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

function getCurrentLabel() {
    var input = document.getElementById('boxLabel');
    return input ? input.value.trim() : '';
}

function setupSegmentation() {
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
        var pos = mousePos(e, this);
        g_hoverX = pos.x;
        g_hoverY = pos.y;
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
            return;
        }
        if (g_resize) {
            resizeBox(g_currentBox, xDiff, yDiff);
        }
    });

    $('#canvas').mouseup(function(e) {
        g_move = false;
        g_resize = false;
        if (!g_paint) return;
        g_paint = false;
        g_annotationHasChanged = true;
        addBox(g_currentFrameNr, g_BBx, g_BBy, g_BBx2, g_BBy2, getCurrentLabel());
    });

    $('#canvas').mouseleave(function(e) {
        if (g_paint) {
            g_annotationHasChanged = true;
            addBox(g_currentFrameNr, g_BBx, g_BBy, g_BBx2, g_BBy2, getCurrentLabel());
            redrawSequence();
            g_paint = false;
        }
        g_hoverX = null;
        g_hoverY = null;
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

    $('#clearButton').click(function() {
        g_annotationHasChanged = true;
        g_boxes = {};
        $('#slider').slider('value', g_frameNr);
        redrawSequence();
    });

    // Ctrl+C: copy current frame's boxes to the next frame
    $(document).keydown(function(e) {
        if (e.ctrlKey && e.which === 67) {
            var tag = e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA') // Don't hijack normal text copy (e.g. boxLabel field)
                return;
            e.preventDefault();
            copyToNext();
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

function removeBox(boxNr) {
    g_boxes[g_currentFrameNr].splice(boxNr, 1);
    g_annotationHasChanged = true;
    redrawSequence();
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

function addBox(frame_nr, x, y, x2, y2, label, color) {
    if (Math.abs(x2 - x) > g_minimumSize && Math.abs(y2 - y) > g_minimumSize) {
        if (labelExistsInFrame(frame_nr, label)) return;
        if (!color) color = stringToColor(label);
        else if (label && !g_labelColorMap[label]) g_labelColorMap[label] = color;
        var box = createBox(x, y, x2, y2, label, color);
        if (!(frame_nr in g_boxes))
            g_boxes[frame_nr] = [];
        g_boxes[frame_nr].push(box);
        addKeyFrame(frame_nr);
        redrawSequence();
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

function copyToNext() {
    if (g_currentFrameNr < g_sequenceLength + 1) {
        var boxes_to_copy = g_boxes[g_currentFrameNr];
        if (!boxes_to_copy || boxes_to_copy.length === 0) return;
        for (var i = 0; i < boxes_to_copy.length; i++) {
            var b = boxes_to_copy[i];
            addBox(
                g_currentFrameNr + 1,
                b.x, b.y,
                b.x + b.width,
                b.y + b.height,
                b.label,
                b.color  // preserve color
            );
        }
    }
}

function loadBBTask(image_sequence_id) {
    g_backgroundImage = new Image();
    g_backgroundImage.src = '/show_frame/' + image_sequence_id + '/' + 0 + '/' + g_taskID + '/';
    g_backgroundImage.onload = function() {
        g_canvasWidth = this.width;
        g_canvasHeight = this.height;
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
