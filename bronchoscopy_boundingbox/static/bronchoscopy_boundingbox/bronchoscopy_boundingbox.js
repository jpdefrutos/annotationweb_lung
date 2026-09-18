var g_backgroundImage;
var g_paint = false;
var g_subsequenceLabels = {};
var g_frameNr;
var g_BBx;
var g_BBy;
var g_BBx2;
var g_BBy2;
var g_boxes = {};
var g_minimumSize = 10;
var g_move = false;
var g_resize = false;
var g_invalidBoxNr = 999999;
var g_currentBox = g_invalidBoxNr;
var g_cornerSize = 20;
var g_hoverX = null;
var g_hoverY = null;
var g_undoStack = []; // {type: 'add'|'remove', frame_nr, box, index}
var g_hydrationBoxes = []; // Saved boxes from the server, applied once real canvas size is known
var g_boxClipboard = []; // Boxes saved via copyAllBoxes(), for pasting onto a non-adjacent frame

function getCurrentLabel() {
    var input = document.getElementById('boxLabel');
    return input ? input.value.trim() : '';
}

// Position clamped to the image bounds, so dragging the mouse past the edge of
// the frame (which happens often when a box needs to reach the frame's actual
// edge) still resolves to a position exactly on that edge, instead of requiring
// the cursor to land on the exact last pixel of the image.
function clampedMousePos(e, canvas) {
    var pos = mousePos(e, canvas);
    pos.x = clamp(pos.x, 0, g_canvasWidth);
    pos.y = clamp(pos.y, 0, g_canvasHeight);
    return pos;
}

function applyDragMove(pos) {
    if (g_paint) {
        g_BBx2 = pos.x;
        g_BBy2 = pos.y;
        redrawSequence();
        return;
    }
    var xDiff = pos.x - g_BBx;
    var yDiff = pos.y - g_BBy;
    g_BBx = pos.x;
    g_BBy = pos.y;
    if (g_move) {
        moveBox(g_currentBox, xDiff, yDiff);
    } else if (g_resize) {
        resizeBox(g_currentBox, xDiff, yDiff);
    }
}

function finishDrag() {
    if (g_move || g_resize) {
        g_move = false;
        g_resize = false;
        return;
    }
    if (g_paint) {
        g_paint = false;
        g_annotationHasChanged = true;
        addBox(g_currentFrameNr, g_BBx, g_BBy, g_BBx2, g_BBy2, getCurrentLabel());
    }
}

function setupSegmentation() {
    var canvas = document.getElementById('canvas');

    $('#canvas').mousedown(function(e) {
        var pos = mousePos(e, this);
        g_BBx = pos.x;
        g_BBy = pos.y;
        var inside = isInsideBox(pos.x, pos.y);
        if (inside.isInside) {
            g_currentBox = inside.boxNr;
            if (inside.isInsideCorner)
                g_resize = true;
            else
                g_move = true;
            return;
        }
        if (!getCurrentLabel()) {
            var el = document.getElementById('boxLabel');
            if (el) { el.style.outline = '2px solid red'; setTimeout(function(){ el.style.outline = ''; }, 1000); }
            return;
        }
        g_paint = true;
    });

    $('#canvas').mousemove(function(e) {
        var pos = clampedMousePos(e, this);
        g_hoverX = pos.x;
        g_hoverY = pos.y;
        applyDragMove(pos);
    });

    $('#canvas').mouseup(function(e) {
        finishDrag();
    });

    $('#canvas').mouseleave(function(e) {
        g_hoverX = null;
        g_hoverY = null;
    });

    // The handlers above only fire while the cursor is over the canvas, so
    // dragging past the edge of the frame would otherwise freeze the box at
    // whatever position was last inside the canvas. These document-level
    // fallbacks keep an in-progress paint/move/resize going (clamped to the
    // image bounds) once the cursor leaves the canvas, and let releasing the
    // mouse anywhere - not just back over the canvas - end the drag.
    $(document).mousemove(function(e) {
        if (e.target === canvas) return; // already handled above
        if (!g_paint && !g_move && !g_resize) return;
        applyDragMove(clampedMousePos(e, canvas));
    });

    $(document).mouseup(function(e) {
        if (e.target === canvas) return; // already handled above
        finishDrag();
    });

    // A drag released outside the browser window (e.g. alt-tab) never fires
    // mouseup at all; drop it instead of leaving g_paint/g_move/g_resize stuck.
    $(window).on('blur', function() {
        g_paint = false;
        g_move = false;
        g_resize = false;
    });

    $('#canvas').dblclick(function(e) {
        var pos = mousePos(e, this);
        var inside = isInsideBox(pos.x, pos.y);
        if (inside.isInside)
            removeBox(inside.boxNr);
    });

    // Ctrl+D: delete the box under the mouse pointer (alternative to double-click)
    $(document).keydown(function(e) {
        if (e.ctrlKey && e.which === 68) {
            var tag = e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA')
                return;
            e.preventDefault();
            if (g_hoverX === null || g_hoverY === null)
                return;
            var inside = isInsideBox(g_hoverX, g_hoverY);
            if (inside.isInside)
                removeBox(inside.boxNr);
        }
    });

    // Ctrl+Z: undo the last added or deleted box
    $(document).keydown(function(e) {
        if (e.ctrlKey && e.which === 90) {
            var tag = e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA')
                return;
            e.preventDefault();
            undoLastBoxAction();
        }
    });

    $('#clearButton').click(function() {
        g_annotationHasChanged = true;
        g_boxes = {};
        g_undoStack = [];
        $('#slider').slider('value', g_frameNr);
        redrawSequence();
        rebuildLabelDropdown();
    });

    // Selecting an existing label loads it into the Box label field so it can be
    // tweaked (e.g. picking "1,2,1,1" to then edit into "1,2,1,2") rather than
    // retyped from scratch. Select the text so retyping the differing part is a
    // single keystroke away.
    $('#usedLabelsSelect').change(function() {
        var val = $(this).val();
        if (!val) return;
        var boxLabelInput = document.getElementById('boxLabel');
        boxLabelInput.value = val;
        boxLabelInput.focus();
        boxLabelInput.select();
    });

    $('#renameLabelButton').click(function() {
        renameSelectedLabel();
    });

    rebuildLabelDropdown();

    // Ctrl+C: copy current frame's boxes to the next frame
    // Ctrl+Shift+C: copy current frame's boxes to the previous frame
    $(document).keydown(function(e) {
        if (e.ctrlKey && e.which === 67) {
            var tag = e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA') // Don't hijack normal text copy (e.g. boxLabel field)
                return;
            e.preventDefault();
            if (e.shiftKey)
                copyToPrevious();
            else
                copyToNext();
        }
    });

    // Ctrl+Shift+A: copy all boxes on the current frame to the clipboard
    // Ctrl+Shift+V: paste all boxes from the clipboard onto the current frame
    // (avoids Alt-based combos: Ctrl+Alt is indistinguishable from AltGr on many
    // European keyboard layouts and would misfire there)
    $(document).keydown(function(e) {
        if (e.ctrlKey && e.shiftKey && (e.which === 65 || e.which === 86)) {
            var tag = e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA')
                return;
            e.preventDefault();
            if (e.which === 65)
                copyAllBoxes();
            else
                pasteAllBoxes();
        }
    });

    try { redrawSequence(); } catch(e) {}
}

function isInsideBox(x, y) {
    var boxNr = g_invalidBoxNr;
    var isInside = false;
    var isInsideCorner = false;
    if (g_currentFrameNr in g_boxes) {
        for (var i = 0; i < g_boxes[g_currentFrameNr].length; ++i) {
            var box = g_boxes[g_currentFrameNr][i];
            if (x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height) {
                isInside = true;
                if (!isInsideCorner)
                    boxNr = i;
                if (x >= box.x + box.width - g_cornerSize && y >= box.y + box.height - g_cornerSize)
                    isInsideCorner = true;
            }
        }
    }
    return { isInside: isInside, boxNr: boxNr, isInsideCorner: isInsideCorner };
}

function removeBox(boxNr, updateDropdown) {
    if (updateDropdown === undefined) updateDropdown = true;
    var frame_nr = g_currentFrameNr;
    var removed = g_boxes[frame_nr].splice(boxNr, 1)[0];
    if (removed)
        g_undoStack.push({type: 'remove', frame_nr: frame_nr, box: removed, index: boxNr});
    g_annotationHasChanged = true;
    redrawSequence();
    if (updateDropdown)
        rebuildLabelDropdown();
}

// Deletes every box on the current frame, one at a time via removeBox() so each
// is pushed onto the undo stack individually - Ctrl+Z undoes them one by one,
// same as deleting them by hand. The dropdown is only rebuilt once at the end
// rather than after each removal, since rebuildLabelDropdown() rescans every
// box on every frame and doing that per-box makes bulk deletion quadratic.
function deleteAllBoxesInFrame() {
    var boxes = g_boxes[g_currentFrameNr];
    if (!boxes || boxes.length === 0) return;
    if (!confirm('Delete all ' + boxes.length + ' box(es) on this frame?')) return;
    for (var i = boxes.length - 1; i >= 0; i--) {
        removeBox(i, false);
    }
    rebuildLabelDropdown();
}

function undoLastBoxAction() {
    var action = g_undoStack.pop();
    if (!action) return;
    if (!(action.frame_nr in g_boxes))
        g_boxes[action.frame_nr] = [];
    if (action.type === 'add') {
        var idx = g_boxes[action.frame_nr].indexOf(action.box);
        if (idx !== -1)
            g_boxes[action.frame_nr].splice(idx, 1);
    } else if (action.type === 'remove') {
        var insertAt = Math.min(action.index, g_boxes[action.frame_nr].length);
        g_boxes[action.frame_nr].splice(insertAt, 0, action.box);
    }
    g_annotationHasChanged = true;
    if (action.frame_nr === g_currentFrameNr)
        redrawSequence();
    rebuildLabelDropdown();
}

// Numeric-alphabetical sort of comma-separated segments, compared segment by
// segment as numbers rather than as plain strings, so e.g. "1,2" < "1,10"
// (a plain string compare would put "1,10" first). A label that is an exact
// prefix of another (e.g. "1" vs "1,1") sorts first.
// e.g. "1" < "1,1" < "1,2" < "1,2,1" < "1,10" < "2"
function compareLabels(a, b) {
    var as = a.split(',');
    var bs = b.split(',');
    var len = Math.min(as.length, bs.length);
    for (var i = 0; i < len; i++) {
        var an = parseInt(as[i], 10);
        var bn = parseInt(bs[i], 10);
        if (!isNaN(an) && !isNaN(bn) && an !== bn) return an - bn;
        if (as[i] !== bs[i]) return as[i] < bs[i] ? -1 : 1;
    }
    return as.length - bs.length;
}

function getUsedLabels() {
    var seen = {};
    for (var frame_nr in g_boxes) {
        var boxesInFrame = g_boxes[frame_nr];
        for (var i = 0; i < boxesInFrame.length; i++) {
            if (boxesInFrame[i].label)
                seen[boxesInFrame[i].label] = true;
        }
    }
    return Object.keys(seen).sort(compareLabels);
}

// The "Used labels" dropdown is just a shortcut to fill in the Box label field
// (see the change handler above) - it doesn't gate what can be drawn. The Box
// label field always accepts free text, and drawing a box with a label that
// hasn't been used before on this video simply creates it, picking up a color
// from stringToColor() the same way any other new label does.
function rebuildLabelDropdown() {
    var select = document.getElementById('usedLabelsSelect');
    if (!select) return;
    var currentValue = select.value;
    var labels = getUsedLabels();

    select.innerHTML = '';
    for (var i = 0; i < labels.length; i++) {
        var option = document.createElement('option');
        option.value = labels[i];
        option.textContent = labels[i];
        select.appendChild(option);
    }

    if (labels.indexOf(currentValue) !== -1) {
        select.value = currentValue;
    }
}

// Renames the label currently selected in the dropdown on every box that has it,
// across every frame of the current video. If a frame already has a box with the
// target name, that frame's box is left under the old name to avoid a duplicate.
function renameSelectedLabel() {
    var select = document.getElementById('usedLabelsSelect');
    if (!select) return;
    var oldLabel = select.value;
    if (!oldLabel) {
        alert('Select an existing label to rename.');
        return;
    }

    var newLabel = prompt('Rename label "' + oldLabel + '" to:', oldLabel);
    if (newLabel === null) return; // Cancelled
    newLabel = newLabel.trim();
    if (!newLabel || newLabel === oldLabel) return;

    var existingCount = 0;
    for (var fn in g_boxes) {
        var boxesInFn = g_boxes[fn];
        for (var bi = 0; bi < boxesInFn.length; bi++) {
            if (boxesInFn[bi].label === newLabel) existingCount++;
        }
    }
    if (existingCount > 0) {
        var proceed = confirm(
            '"' + newLabel + '" is already used on ' + existingCount + ' other box(es) in this video.\n\n' +
            'Merging is permanent — you won\'t be able to tell old "' + newLabel + '" boxes apart ' +
            'from the renamed ones afterward.\n\n' +
            'Tip: if you also want the existing "' + newLabel + '" boxes to become something else, ' +
            'rename those first.\n\n' +
            'Continue and merge anyway?'
        );
        if (!proceed) return;
    }

    var newColor = stringToColor(newLabel);
    var skippedFrames = [];
    for (var frame_nr in g_boxes) {
        var boxesInFrame = g_boxes[frame_nr];
        var targetExists = false;
        for (var i = 0; i < boxesInFrame.length; i++) {
            if (boxesInFrame[i].label === newLabel) { targetExists = true; break; }
        }
        for (var k = 0; k < boxesInFrame.length; k++) {
            if (boxesInFrame[k].label !== oldLabel) continue;
            if (targetExists) {
                skippedFrames.push(frame_nr);
                continue;
            }
            boxesInFrame[k].label = newLabel;
            boxesInFrame[k].color = newColor;
        }
    }

    g_annotationHasChanged = true;
    rebuildLabelDropdown();
    select.value = newLabel;
    $('#boxLabel').val(newLabel);
    redrawSequence();

    if (skippedFrames.length > 0) {
        alert('Renamed everywhere except frame(s) ' + skippedFrames.join(', ') +
            ', which already had a box labeled "' + newLabel + '". Those were left as "' + oldLabel + '".');
    }
}

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(v, hi));
}

function moveBox(boxNr, xDiff, yDiff) {
    var box = g_boxes[g_currentFrameNr][boxNr];
    box.x = clamp(box.x + xDiff, 0, g_canvasWidth - box.width);
    box.y = clamp(box.y + yDiff, 0, g_canvasHeight - box.height);
    redrawSequence();
}

function resizeBox(boxNr, xDiff, yDiff) {
    var box = g_boxes[g_currentFrameNr][boxNr];
    if (box.width > -xDiff + g_minimumSize)
        box.width = clamp(box.width + xDiff, g_minimumSize, g_canvasWidth - box.x);
    if (box.height > -yDiff + g_minimumSize)
        box.height = clamp(box.height + yDiff, g_minimumSize, g_canvasHeight - box.y);
    redrawSequence();
}

function createBox(x, y, x2, y2, label, color) {
    x = clamp(x, 0, g_canvasWidth);
    y = clamp(y, 0, g_canvasHeight);
    x2 = clamp(x2, 0, g_canvasWidth);
    y2 = clamp(y2, 0, g_canvasHeight);
    var originX = Math.min(x, x2);
    var originY = Math.min(y, y2);
    return {
        x: originX,
        y: originY,
        width: Math.max(x, x2) - originX,
        height: Math.max(y, y2) - originY,
        label: label,
        color: color
    };
}

function addBox(frame_nr, x, y, x2, y2, label, color, recordUndo, updateDropdown) {
    if (recordUndo === undefined) recordUndo = true;
    if (updateDropdown === undefined) updateDropdown = true;
    if (Math.abs(x2 - x) > g_minimumSize && Math.abs(y2 - y) > g_minimumSize) {
        if (labelExistsInFrame(frame_nr, label)) return;
        if (!color) color = stringToColor(label);
        else if (label && !g_labelColorMap[label]) g_labelColorMap[label] = color;
        var box = createBox(x, y, x2, y2, label, color);
        if (!(frame_nr in g_boxes))
            g_boxes[frame_nr] = [];
        g_boxes[frame_nr].push(box);
        // Boxes hydrated from the server on page load are not undoable user
        // actions - recording them here would let Ctrl+Z silently delete
        // already-saved boxes instead of just the most recent new one.
        if (recordUndo)
            g_undoStack.push({type: 'add', frame_nr: frame_nr, box: box});
        addKeyFrame(frame_nr);
        redrawSequence();
        // rebuildLabelDropdown() rescans every box on every frame; callers that
        // add many boxes in a loop (hydration, copy/paste) pass false and rebuild
        // once after the loop instead, to avoid an O(n^2) stall.
        if (updateDropdown)
            rebuildLabelDropdown();
    }
}

function labelExistsInFrame(frame_nr, label) {
    if (!(frame_nr in g_boxes)) return false;
    for (var i = 0; i < g_boxes[frame_nr].length; i++) {
        if (g_boxes[frame_nr][i].label === label) return true;
    }
    return false;
}

function redraw() {
    // Draw in-progress box (only if label not already used in this frame)
    if (g_paint && getCurrentLabel() && !labelExistsInFrame(g_currentFrameNr, getCurrentLabel())) {
        var previewColor = stringToColor(getCurrentLabel());
        var preview = createBox(g_BBx, g_BBy, g_BBx2, g_BBy2, '', previewColor);
        g_context.beginPath();
        g_context.lineWidth = 2;
        g_context.strokeStyle = previewColor;
        g_context.rect(preview.x, preview.y, preview.width, preview.height);
        g_context.stroke();
    }

    if (!(g_currentFrameNr in g_boxes)) return;

    for (var i = 0; i < g_boxes[g_currentFrameNr].length; ++i) {
        var box = g_boxes[g_currentFrameNr][i];
        g_context.beginPath();
        g_context.lineWidth = 2;
        g_context.strokeStyle = box.color;
        g_context.rect(box.x, box.y, box.width, box.height);
        // Resize corner indicator
        g_context.moveTo(box.x + box.width - g_cornerSize, box.y + box.height);
        g_context.lineTo(box.x + box.width, box.y + box.height - g_cornerSize);
        g_context.stroke();
        // Label text
        if (box.label) {
            g_context.font = 'bold 13px sans-serif';
            g_context.fillStyle = box.color;
            g_context.fillText(box.label, box.x + 4, box.y + 16);
        }
    }
}

function redrawSequence() {
    var index = g_currentFrameNr - g_startFrame;
    g_context.drawImage(g_sequence[index], 0, 0, g_canvasWidth, g_canvasHeight);
    redraw();
    var label = String(g_currentFrameNr) in g_subsequenceLabels
        ? g_subsequenceLabels[String(g_currentFrameNr)] : 'N/A';
    $('#subsequenceLabel').text(label);
}

// goToFrame() (annotationweb.js) clamps to g_framesLoaded-1, i.e. how many of this
// sequence's frame images have *finished downloading so far* - not a sequence
// boundary. On a "show_entire_sequence" task with ~1800 frames, that count lags
// well behind the real position for a while after page load, so goToFrame() would
// silently land on whatever frame happened to be loaded instead of the one we just
// copied to. Navigate directly to the real target frame instead, redrawing once its
// image has actually finished loading if it hasn't yet.
function goToCopiedFrame(frameNr) {
    setPlayButton(false);
    g_currentFrameNr = frameNr;
    $('#slider').slider('value', frameNr);
    $('#currentFrame').text(g_currentFrameNr);
    var marker_index = g_targetFrames.findIndex(index => index === frameNr);
    if (marker_index) {
        g_currentTargetFrameIndex = g_currentFrameNr;
    } else {
        g_currentTargetFrameIndex = -1;
    }

    var img = g_sequence[frameNr - g_startFrame];
    if (img && img.complete && img.naturalWidth > 0) {
        redrawSequence();
    } else if (img) {
        img.addEventListener('load', function onLoaded() {
            img.removeEventListener('load', onLoaded);
            if (g_currentFrameNr === frameNr) redrawSequence();
        });
    }
}

function copyToNext() {
    if (g_currentFrameNr < g_startFrame + g_sequenceLength) {
        var boxes_to_copy = g_boxes[g_currentFrameNr];
        if (!boxes_to_copy || boxes_to_copy.length === 0) return;
        var nextFrameNr = g_currentFrameNr + 1;
        for (var i = 0; i < boxes_to_copy.length; i++) {
            var b = boxes_to_copy[i];
            addBox(
                nextFrameNr,
                b.x, b.y,
                b.x + b.width,
                b.y + b.height,
                b.label,
                b.color,  // preserve color
                true, false
            );
        }
        rebuildLabelDropdown();
        g_annotationHasChanged = true;
        goToCopiedFrame(nextFrameNr);
    }
}

function copyToPrevious() {
    if (g_currentFrameNr > g_startFrame) {
        var boxes_to_copy = g_boxes[g_currentFrameNr];
        if (!boxes_to_copy || boxes_to_copy.length === 0) return;
        var previousFrameNr = g_currentFrameNr - 1;
        for (var i = 0; i < boxes_to_copy.length; i++) {
            var b = boxes_to_copy[i];
            addBox(
                previousFrameNr,
                b.x, b.y,
                b.x + b.width,
                b.y + b.height,
                b.label,
                b.color,  // preserve color
                true, false
            );
        }
        rebuildLabelDropdown();
        g_annotationHasChanged = true;
        goToCopiedFrame(previousFrameNr);
    }
}

function updateBoxClipboardStatus() {
    var el = document.getElementById('boxClipboardStatus');
    if (!el) return;
    el.textContent = g_boxClipboard.length > 0
        ? 'Clipboard: ' + g_boxClipboard.length + ' box' + (g_boxClipboard.length === 1 ? '' : 'es')
        : 'Clipboard: empty';
}

// Copies every box on the current frame into a clipboard that survives jumping
// to a non-adjacent frame (unlike copyToNext/copyToPrevious), for the case where
// the same set of boxes needs to reappear several frames later.
function copyAllBoxes() {
    var boxes = g_boxes[g_currentFrameNr];
    if (!boxes || boxes.length === 0) {
        alert('No boxes on the current frame to copy.');
        return;
    }
    g_boxClipboard = boxes.map(function(b) {
        return { x: b.x, y: b.y, width: b.width, height: b.height, label: b.label, color: b.color };
    });
    updateBoxClipboardStatus();
}

function pasteAllBoxes() {
    if (!g_boxClipboard || g_boxClipboard.length === 0) {
        alert("Clipboard is empty. Use 'Copy all boxes' first.");
        return;
    }
    for (var i = 0; i < g_boxClipboard.length; i++) {
        var b = g_boxClipboard[i];
        addBox(
            g_currentFrameNr,
            b.x, b.y,
            b.x + b.width,
            b.y + b.height,
            b.label,
            b.color,  // preserve color
            true, false
        );
    }
    rebuildLabelDropdown();
    g_annotationHasChanged = true;
}

function loadBBTask(image_sequence_id) {
    g_backgroundImage = new Image();
    g_backgroundImage.src = '/show_frame/' + image_sequence_id + '/' + 0 + '/' + g_taskID + '/';
    g_backgroundImage.onload = function() {
        g_canvasWidth = this.width;
        g_canvasHeight = this.height;

        // Hydrate saved boxes now that the real canvas size is known. Doing this
        // earlier (e.g. synchronously at page load) would clamp coordinates against
        // the 512x512 default in createBox(), collapsing/mispositioning any box
        // outside that range.
        for (var i = 0; i < g_hydrationBoxes.length; i++) {
            var b = g_hydrationBoxes[i];
            try {
                addBox(b.frame_nr, b.x, b.y, b.x + b.width, b.y + b.height, b.label, undefined, false, false);
            } catch (e) {}
        }
        g_hydrationBoxes = [];
        rebuildLabelDropdown();

        // Snap to the first key frame before setting up mouse handlers.
        // loadSequence sets g_currentFrameNr=0 because g_targetFrames is empty
        // at that point in its code; addKeyFrame runs later, so we correct it here.
        if (g_targetFrames.length > 0) {
            g_currentFrameNr = g_targetFrames[0];
            $('#slider').slider('value', g_currentFrameNr);
            $('#currentFrame').text(g_currentFrameNr);
        }
        setupSegmentation();
    };
}

function sendDataForSave() {
    return $.ajax({
        type: 'POST',
        url: '/bronchoscopy-bounding-box/save/',
        data: {
            image_id: g_imageID,
            boxes: JSON.stringify(g_boxes),
            task_id: g_taskID,
            target_frames: JSON.stringify(g_targetFrames),
            quality: $('input[name=quality]:checked').val(),
            rejected: g_rejected ? 'true' : 'false',
            comments: $('#comments').val(),
        },
        dataType: 'json'
    });
}
