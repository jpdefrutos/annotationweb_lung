// Automatic box suggestions (lumen proposals) for the bronchoscopy bounding box task.
//
// The server (/bronchoscopy-bounding-box/propose/) detects dark airway openings
// and, with tracking on, gives the same lumen the same box_id over consecutive
// frames. Suggestions live only in the browser (g_proposals) and are drawn
// dashed; they are never saved. Accepting one turns it into an ordinary box in
// g_boxes, which is saved the normal way.
//
// box_id is only unique within one run (a new run starts again at 1), so a
// track is always identified by (run_id, box_id), never by box_id alone.

var g_imageSequenceID = null;
var g_proposals = {};          // frame_nr -> [proposal]
var g_proposalState = null;    // tracker state from the last response, sent back with the next chunk
var g_selectedProposal = null;
var g_proposalBusy = false;
var g_overlapHideIou = 0.5;    // hide a suggestion that overlaps an accepted box this much
var g_trackLetters = {};       // track key -> letter, so a track keeps its letter over frames
var g_letterCounters = {};     // counter key -> how many letters handed out

// ── settings and requests ────────────────────────────────────────────────────

function proposalParams() {
    var levels = parseInt($('#propLevels').val(), 10);
    var track = $('#propTrack').is(':checked');
    var p = {
        levels: levels,
        min_area: $('#propMinArea').val(),
        track: track,
    };
    if (levels === 2) {
        p.min_gap = $('#propMinGap').val();
        p.child_min = $('#propChildMin').val();
    }
    if (track) {
        p.track_iou = $('#propTrackIou').val();
        p.max_lost = $('#propMaxLost').val();
        p.min_track_len = $('#propMinTrackLen').val();
        if (levels === 2)
            p.track_iou2 = $('#propTrackIou2').val();
    }
    return p;
}

function updateProposalSettingsVisibility() {
    var levels2 = $('#propLevels').val() === '2';
    $('#proposalPanel .prop-level2').toggle(levels2);
    $('#proposalPanel .prop-track').toggle($('#propTrack').is(':checked'));
}

function setProposalStatus(text, isError) {
    $('#proposalStatus').text(text).css('color', isError ? '#ff6060' : '');
}

function requestProposals(startFrame) {
    if (g_proposalBusy || g_imageSequenceID === null) return;
    var count = parseInt($('#propCount').val(), 10) || 60;
    g_proposalBusy = true;
    $('#proposeButton').prop('disabled', true);
    setProposalStatus('Finding boxes in frames ' + startFrame + '–' + (startFrame + count - 1) +
        '… (first run on a video also estimates the scope area)');

    $.ajax({
        type: 'POST',
        url: '/bronchoscopy-bounding-box/propose/',
        data: {
            task_id: g_taskID,
            image_sequence_id: g_imageSequenceID,
            start: startFrame,
            count: count,
            params: JSON.stringify(proposalParams()),
            state: g_proposalState ? JSON.stringify(g_proposalState) : '',
        },
        dataType: 'json'
    }).done(function(data) {
        if (!data.success) {
            setProposalStatus(data.message || 'Suggestion failed', true);
            return;
        }
        var continued = g_proposalState && data.state.run_id === g_proposalState.run_id;
        var first = data.frames[0], last = data.frames[1];
        var prevState = g_proposalState, saved = {};
        for (var f = first; f <= last; f++)
            if (g_proposals[f]) saved[f] = g_proposals[f];
        g_proposalState = data.state;
        var hidden = storeProposals(data.rows, first, last);
        // Ctrl+Z puts back the suggestions these frames had before (if any)
        // and the tracker state, so tracks continue as they did.
        g_undoStack.push({type: 'custom', undo: function() {
            if (g_selectedProposal && g_selectedProposal.frame_nr >= first &&
                    g_selectedProposal.frame_nr <= last)
                selectProposal(null);
            for (var u = first; u <= last; u++) {
                if (saved[u]) g_proposals[u] = saved[u];
                else delete g_proposals[u];
            }
            g_proposalState = prevState;
            refreshChildNames();
            setProposalStatus('Undid the suggestions for frames ' + first + '–' + last + '.');
        }});
        var auto = continued ? autoAcceptContinuedTracks(first, last) : {boxes: 0, tracks: 0};
        var n = data.rows.length - hidden - auto.boxes;
        setProposalStatus(n + ' suggestion' + (n === 1 ? '' : 's') + ' in frames ' +
            data.frames[0] + '–' + data.frames[1] +
            (continued ? ' (tracks continued)' : ' (new run ' + data.state.run_id + ')') + '.' +
            (auto.boxes ? ' ' + auto.boxes + ' box(es) continuing ' + auto.tracks +
                ' accepted track(s) were accepted automatically (Ctrl+Z undoes).' : '') +
            (data.dropped ? ' ' + data.dropped + ' box(es) on short tracks were left out.' : '') +
            (hidden ? ' ' + hidden + ' hidden because they overlap a box you already have.' : ''));
        if (startFrame !== g_currentFrameNr)
            goToCopiedFrame(startFrame);
        else
            redrawSequence();
    }).fail(function(xhr) {
        var msg = xhr.responseJSON && xhr.responseJSON.message
            ? xhr.responseJSON.message : 'Suggestion failed (' + xhr.status + ')';
        setProposalStatus(msg, true);
    }).always(function() {
        g_proposalBusy = false;
        $('#proposeButton').prop('disabled', false);
    });
}

// Names already given to tracks, from frames before firstFrame: the name on
// the latest earlier frame of each (run_id, box_id), taken from unaccepted
// suggestions that were renamed and from boxes accepted from a suggestion.
// Returns {"<run_id>|<box_id>": name}.
function earlierTrackNames(firstFrame) {
    var best = {};   // key -> {frame, name}
    function consider(src, frame, name) {
        var key = src.run_id + '|' + src.box_id;
        if (!best[key] || frame > best[key].frame) best[key] = { frame: frame, name: name };
    }
    for (var f in g_proposals) {
        var fn = parseInt(f, 10);
        if (fn >= firstFrame) continue;
        var list = g_proposals[f];
        for (var i = 0; i < list.length; i++)
            if (list[i].label !== list[i].placeholder) consider(list[i], fn, list[i].label);
    }
    for (var g in g_boxes) {
        var gn = parseInt(g, 10);
        if (gn >= firstFrame) continue;
        var boxes = g_boxes[g];
        for (var j = 0; j < boxes.length; j++)
            if (boxes[j].src) consider(boxes[j].src, gn, boxes[j].label);
    }
    var names = {};
    for (var key in best) names[key] = best[key].name;
    return names;
}

function iou(a, b) {
    var iw = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
    var ih = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
    if (iw <= 0 || ih <= 0) return 0;
    var inter = iw * ih;
    return inter / (a.width * a.height + b.width * b.height - inter);
}

// True if the suggestion overlaps a box already on its frame (accepted, drawn
// by hand or loaded from an earlier save) with IoU >= g_overlapHideIou.
function coveredByBox(p) {
    var boxes = g_boxes[p.frame_nr] || [];
    for (var i = 0; i < boxes.length; i++)
        if (iou(p, boxes[i]) >= g_overlapHideIou) return true;
    return false;
}

// ── suggested names ──────────────────────────────────────────────────────────
//
// A main box is named after the frame's branch from the subsequence
// classification plus a letter: on branch "1,2,1" two openings become
// "1,2,1,a" and "1,2,1,b". A nested box gets its parent's current name plus a
// letter: "1,2,1,a,a", "1,2,1,a,b". The annotator then only replaces the
// letters with numbers. A track keeps its letter over frames; new tracks on
// the same branch (or parent) get the next unused letter, so names on one frame
// never clash. Without tracking, letters are handed out per frame. Frames
// without a subsequence label keep the server's temporary name (T7 / D3).

// 0 -> a, 25 -> z, 26 -> aa, 27 -> ab ...
function letterName(i) {
    var s = '';
    i += 1;
    while (i > 0) {
        var m = (i - 1) % 26;
        s = String.fromCharCode(97 + m) + s;
        i = Math.floor((i - 1) / 26);
    }
    return s;
}

function letterFor(counterKey, trackKey) {
    if (!(trackKey in g_trackLetters)) {
        var n = g_letterCounters[counterKey] || 0;
        g_trackLetters[trackKey] = letterName(n);
        g_letterCounters[counterKey] = n + 1;
    }
    return g_trackLetters[trackKey];
}

function frameBranch(frame_nr) {
    var b = g_subsequenceLabels[String(frame_nr)];
    return b ? String(b) : null;
}

// Current name of the parent of a nested box: the parent suggestion on the
// same frame (renamed or not), or the box accepted from it. Linked by
// (run_id, parent_id), never by name, so renaming cannot break the link.
function parentName(p) {
    if (p.level !== 2 || p.parent_id === null) return null;
    var parent = { run_id: p.run_id, box_id: p.parent_id };
    var list = g_proposals[p.frame_nr] || [];
    for (var i = 0; i < list.length; i++)
        if (sameTrack(list[i], parent)) return list[i].label;
    var boxes = g_boxes[p.frame_nr] || [];
    for (var j = 0; j < boxes.length; j++)
        if (boxes[j].src && sameTrack(boxes[j].src, parent)) return boxes[j].label;
    return null;
}

// Sets r.letter and returns the suggested name. parentLabels holds the names
// of this response's main boxes ("frame|box_id" -> name), which may be hidden
// and so not findable through parentName().
function suggestedName(r, parentLabels, ownName) {
    var tracked = r.label.charAt(0) === 'T';
    // A handed-over box normally takes its track's earlier name (see
    // syncHandedOverNames), so it only gets a letter of its own if it has none.
    if (r.was_nested && r.level === 1 && !ownName) return r.label;
    var scope = r.run_id + (tracked ? '' : '|f' + r.frame_nr);
    var trackKey = scope + '|' + r.box_id;
    if (r.level === 1) {
        var branch = frameBranch(r.frame_nr);
        if (!branch) return r.label;
        r.letter = letterFor(scope + '|branch|' + branch, branch + '|' + trackKey);
        return branch + ',' + r.letter;
    }
    var parent = parentLabels[r.frame_nr + '|' + r.parent_id] || parentName(r) || ('T' + r.parent_id);
    r.letter = letterFor(scope + '|parent|' + r.parent_id, trackKey);
    return parent + ',' + r.letter;
}

// After a parent is renamed or accepted, its still-unnamed children follow:
// parent "1,2,1,1" turns child "1,2,1,a,b" into "1,2,1,1,b".
function refreshChildNames() {
    for (var f in g_proposals) {
        var list = g_proposals[f];
        for (var i = 0; i < list.length; i++) {
            var c = list[i];
            if (c.level !== 2 || c.label !== c.placeholder || !c.letter) continue;
            var pn = parentName(c);
            if (!pn) continue;
            c.label = c.placeholder = pn + ',' + c.letter;
        }
    }
    syncHandedOverNames();
}

// A nested box that became a main box (the server marks it was_nested, e.g.
// when the scope passes its parent) keeps its track and so its name: while
// still unrenamed it takes the name of the track's latest earlier suggestion
// or accepted box. Walks frames in order so the name carries through.
function syncHandedOverNames() {
    var frames = {};
    for (var f in g_proposals) frames[f] = true;
    for (var g in g_boxes) frames[g] = true;
    var order = Object.keys(frames).map(Number).sort(function(a, b) { return a - b; });
    var last = {};      // "run|box" -> {label, placeholder}
    for (var i = 0; i < order.length; i++) {
        var fr = order[i];
        var boxes = g_boxes[fr] || [];
        for (var j = 0; j < boxes.length; j++)
            if (boxes[j].src)
                last[boxes[j].src.run_id + '|' + boxes[j].src.box_id] = {label: boxes[j].label, placeholder: null};
        var list = g_proposals[fr] || [];
        for (var k = 0; k < list.length; k++) {
            var p = list[k], key = p.run_id + '|' + p.box_id;
            if (p.was_nested && p.level === 1 && p.label === p.placeholder) {
                if (last[key]) {
                    p.label = last[key].label;
                    if (last[key].placeholder !== null) p.placeholder = last[key].placeholder;
                } else if (/^T\d+$/.test(p.label)) {
                    p.label = p.placeholder = suggestedName(p, {}, true);   // no earlier name
                }
            }
            last[key] = {label: p.label, placeholder: p.placeholder};
        }
    }
}

// New suggestions for a frame replace any older, unaccepted ones on it, so
// two runs never show overlapping placeholder names on the same frame. A
// suggestion whose track was already named in an earlier frame (e.g. in the
// previous chunk) gets that name, so a track is only named once. Suggestions
// on top of an existing box are not shown. Returns how many were hidden.
function storeProposals(rows, firstFrame, lastFrame) {
    var hidden = 0;
    var names = earlierTrackNames(firstFrame);
    var parentLabels = {};
    for (var f = firstFrame; f <= lastFrame; f++)
        delete g_proposals[f];
    for (var i = 0; i < rows.length; i++) {
        var r = rows[i];
        var tracked = r.label.charAt(0) === 'T';
        r.placeholder = suggestedName(r, parentLabels);
        r.label = r.placeholder;
        var key = r.run_id + '|' + r.box_id;
        if (tracked && key in names) r.label = names[key];
        if (r.level === 1) parentLabels[r.frame_nr + '|' + r.box_id] = r.label;
        if (coveredByBox(r)) {
            hidden++;
            continue;
        }
        if (!(r.frame_nr in g_proposals))
            g_proposals[r.frame_nr] = [];
        g_proposals[r.frame_nr].push(r);
    }
    refreshChildNames();
    if (g_selectedProposal && !proposalExists(g_selectedProposal))
        selectProposal(null);
    return hidden;
}

// ── lookup ───────────────────────────────────────────────────────────────────

function proposalExists(p) {
    var list = g_proposals[p.frame_nr];
    return !!list && list.indexOf(p) !== -1;
}

function sameTrack(a, b) {
    return a.run_id === b.run_id && a.box_id === b.box_id;
}

// Proposals in p's track: same (run_id, box_id), optionally from fromFrame on.
function trackProposals(p, fromFrame) {
    var out = [];
    for (var f in g_proposals) {
        if (fromFrame !== undefined && parseInt(f, 10) < fromFrame) continue;
        var list = g_proposals[f];
        for (var i = 0; i < list.length; i++)
            if (sameTrack(list[i], p)) out.push(list[i]);
    }
    out.sort(function(a, b) { return a.frame_nr - b.frame_nr; });
    return out;
}

// The smallest suggestion under (x, y) on the current frame, so a nested
// (level-2) box can be picked even though it lies inside its parent.
function proposalAt(x, y) {
    var list = g_proposals[g_currentFrameNr];
    if (!list) return null;
    var best = null;
    for (var i = 0; i < list.length; i++) {
        var p = list[i];
        if (x >= p.x && x <= p.x + p.width && y >= p.y && y <= p.y + p.height) {
            if (!best || p.width * p.height < best.width * best.height) best = p;
        }
    }
    return best;
}

// ── selection ────────────────────────────────────────────────────────────────

function selectProposal(p) {
    g_selectedProposal = p;
    setSelectedProposalControlsEnabled(!!p);
    if (!p) {
        $('#selectedProposalInfo').html('<i>Click a dashed suggestion box in the image to select it.</i>');
        $('#proposalName').val('');
        redrawSequence();
        return;
    }
    var track = trackProposals(p);
    var info = 'Suggestion <b>' + escapeHtml(p.label) + '</b>, level ' + p.level;
    if (track.length > 1)
        info += ', track in frames ' + track[0].frame_nr + '–' + track[track.length - 1].frame_nr +
            ' (' + track.length + ' boxes)';
    $('#selectedProposalInfo').html(info);

    // Pre-fill the current name. For a suggested name, only its last part (the
    // letter) is selected, so typing "1" turns "1,2,1,a" into "1,2,1,1".
    var input = $('#proposalName');
    input.val(p.label).focus();
    var el = input[0];
    var start = p.label === p.placeholder && p.letter ? p.label.lastIndexOf(',') + 1 : 0;
    el.setSelectionRange(start, p.label.length);
    redrawSequence();
}

// The selected-suggestion controls are always visible (so they can be found),
// but disabled until a suggestion is selected.
function setSelectedProposalControlsEnabled(enabled) {
    $('#selectedProposalPanel').find('input, button').prop('disabled', !enabled);
}

function selectedName() {
    return $('#proposalName').val().trim();
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function(c) {
        return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
    });
}

// ── actions ──────────────────────────────────────────────────────────────────

function removeProposal(p) {
    var list = g_proposals[p.frame_nr];
    if (!list) return;
    var i = list.indexOf(p);
    if (i !== -1) list.splice(i, 1);
    if (list.length === 0) delete g_proposals[p.frame_nr];
    if (g_selectedProposal === p) selectProposal(null);
}

// Turns a suggestion into an ordinary box with the given name and returns it,
// or null if the frame already has a box with that name (or the box is too
// small). recordUndo=false when the caller pushes its own undo step.
function acceptProposalAs(p, name, recordUndo) {
    if (recordUndo === undefined) recordUndo = true;
    var box = addBox(p.frame_nr, p.x, p.y, p.x + p.width, p.y + p.height, name, undefined, recordUndo, false);
    if (!box) return null;
    box.src = { run_id: p.run_id, box_id: p.box_id };   // in memory only; for parentName()
    removeProposal(p);
    g_annotationHasChanged = true;
    return box;
}

function confirmPlaceholder(name, p) {
    if (name !== p.placeholder) return true;
    return confirm('"' + name + '" is still the temporary name. Accept it anyway?');
}

function acceptSelectedProposal() {
    var p = g_selectedProposal;
    if (!p) return;
    var name = selectedName();
    if (!name) { alert('Enter a name first.'); return; }
    if (!confirmPlaceholder(name, p)) return;
    var box = acceptProposalAs(p, name, false);
    if (!box) {
        alert('This frame already has a box named "' + name + '" (or the box is too small).');
        return;
    }
    g_undoStack.push({type: 'custom', undo: function() {
        unacceptTrack([{proposal: p, box: box}], 'Undid Accept: the box is a suggestion again.');
    }});
    refreshChildNames();
    rebuildLabelDropdown();
    redrawSequence();
}

function trackScopeFrame() {
    return $('#proposalFromHere').is(':checked') ? g_currentFrameNr : undefined;
}

function describeRange(list) {
    return list.length + ' box(es) in frames ' + list[0].frame_nr + '–' + list[list.length - 1].frame_nr;
}

// A track can contain a wrong link (another lumen took over the ID after fast
// scope motion), so every track-wide action asks for confirmation with its range.
function renameSelectedTrack() {
    var p = g_selectedProposal;
    if (!p) return;
    var name = selectedName();
    if (!name) { alert('Enter a name first.'); return; }
    var list = trackProposals(p, trackScopeFrame());
    if (list.length === 0) return;
    var oldLabels = list.map(function(q) { return q.label; });
    for (var i = 0; i < list.length; i++)
        list[i].label = name;
    g_undoStack.push({type: 'custom', undo: function() {
        for (var j = 0; j < list.length; j++)
            list[j].label = oldLabels[j];
        refreshChildNames();
        if (g_selectedProposal) selectProposal(g_selectedProposal);
        showProposalNotice('Undid Rename track.');
    }});
    refreshChildNames();
    selectProposal(p);
    showProposalNotice('Renamed ' + describeRange(list) + ' to "' + name + '". Ctrl+Z undoes.');
}

function acceptSelectedTrack() {
    var p = g_selectedProposal;
    if (!p) return;
    var name = selectedName();
    if (!name) { alert('Enter a name first.'); return; }
    if (!confirmPlaceholder(name, p)) return;
    var list = trackProposals(p, trackScopeFrame());
    if (list.length === 0) return;
    // No confirmation: the result is shown in the notice line, and one Ctrl+Z
    // takes the whole track back to suggestions.
    var skipped = [], accepted = [];
    for (var i = 0; i < list.length; i++) {
        var box = acceptProposalAs(list[i], name, false);
        if (box)
            accepted.push({proposal: list[i], box: box});
        else
            skipped.push(list[i].frame_nr);
    }
    if (accepted.length > 0)
        g_undoStack.push({type: 'custom', undo: function() { unacceptTrack(accepted); }});
    refreshChildNames();
    rebuildLabelDropdown();
    redrawSequence();
    showProposalNotice('Accepted ' + accepted.length + ' box(es) as "' + name + '"' +
        (accepted.length ? ' in frames ' + accepted[0].proposal.frame_nr + '–' +
            accepted[accepted.length - 1].proposal.frame_nr : '') + '. Ctrl+Z undoes.' +
        (skipped.length ? ' Not accepted on frame(s) ' + skipped.join(', ') + ': "' + name +
            '" already exists there (or the box is too small); those suggestions are kept.' : ''),
        skipped.length > 0);
}

// When a run continues the previous one, a track the annotator accepted just
// before this run (on the last max_lost + 1 frames before it) is accepted on
// the new frames too, under the same name. One undo step turns them all back
// into suggestions. Returns {boxes, tracks} accepted.
function autoAcceptContinuedTracks(first, last) {
    var lookback = (parseInt($('#propMaxLost').val(), 10) || 0) + 1;
    var accepted = {};      // "run|box" -> name on the latest of those frames
    for (var f = first - lookback; f < first; f++) {
        var boxes = g_boxes[f] || [];
        for (var i = 0; i < boxes.length; i++)
            if (boxes[i].src)
                accepted[boxes[i].src.run_id + '|' + boxes[i].src.box_id] = boxes[i].label;
    }
    if (Object.keys(accepted).length === 0) return {boxes: 0, tracks: 0};
    var done = [], tracks = {};
    for (var g = first; g <= last; g++) {
        var list = (g_proposals[g] || []).slice();
        list.sort(function(a, b) { return a.level - b.level; });   // parents first
        for (var k = 0; k < list.length; k++) {
            var key = list[k].run_id + '|' + list[k].box_id;
            if (!(key in accepted)) continue;
            var box = acceptProposalAs(list[k], accepted[key], false);
            if (box) {
                done.push({proposal: list[k], box: box});
                tracks[key] = true;
            }
        }
    }
    if (done.length > 0) {
        g_undoStack.push({type: 'custom', undo: function() {
            unacceptTrack(done, 'Undid the automatic accept: ' + done.length +
                ' box(es) are suggestions again.');
        }});
        refreshChildNames();
        rebuildLabelDropdown();
    }
    return {boxes: done.length, tracks: Object.keys(tracks).length};
}

// Undo for Accept / Accept track: removes the accepted boxes and turns them
// back into suggestions.
function unacceptTrack(accepted, notice) {
    for (var i = 0; i < accepted.length; i++) {
        var a = accepted[i], f = a.proposal.frame_nr;
        var boxes = g_boxes[f] || [];
        var k = boxes.indexOf(a.box);
        if (k !== -1) boxes.splice(k, 1);
        (g_proposals[f] = g_proposals[f] || []).push(a.proposal);
    }
    refreshChildNames();
    showProposalNotice(notice || ('Undid Accept track: ' + accepted.length + ' box(es) are suggestions again.'));
}

function showProposalNotice(text, important) {
    showNotice('#proposalNotice', text, important);
}

function rejectSelectedTrack() {
    var p = g_selectedProposal;
    if (!p) return;
    var list = trackProposals(p, trackScopeFrame());
    if (list.length === 0) return;
    for (var i = 0; i < list.length; i++)
        removeProposal(list[i]);
    g_undoStack.push({type: 'custom', undo: function() {
        for (var j = 0; j < list.length; j++) {
            var f = list[j].frame_nr;
            (g_proposals[f] = g_proposals[f] || []).push(list[j]);
        }
        refreshChildNames();
        showProposalNotice('Undid Reject track: ' + list.length + ' suggestion(s) are back.');
    }});
    redrawSequence();
    showProposalNotice('Rejected ' + describeRange(list) + '. Ctrl+Z undoes.');
}

function rejectProposal(p) {
    removeProposal(p);
    g_undoStack.push({type: 'custom', undo: function() {
        (g_proposals[p.frame_nr] = g_proposals[p.frame_nr] || []).push(p);
        refreshChildNames();
    }});
    redrawSequence();
}

// The suggestion a click at (x, y) should go to, or null if it belongs to the
// box under the cursor (`inside` from isInsideBox). The smaller one wins, so a
// nested suggestion inside an accepted parent can be clicked directly, while
// the rest of the parent still moves it. The box's resize corner always wins.
function proposalUnderCursor(x, y, inside) {
    var p = proposalAt(x, y);
    if (!p || !inside.isInside) return p;
    if (inside.isInsideCorner) return null;
    var box = g_boxes[g_currentFrameNr][inside.boxNr];
    return p.width * p.height < box.width * box.height ? p : null;
}

function clearProposals() {
    var n = 0;
    for (var f in g_proposals) n += g_proposals[f].length;
    if (n === 0) return;
    var saved = g_proposals, savedState = g_proposalState;
    g_proposals = {};
    g_proposalState = null;     // the next run starts fresh
    selectProposal(null);
    g_undoStack.push({type: 'custom', undo: function() {
        g_proposals = saved;
        g_proposalState = savedState;
        refreshChildNames();
        setProposalStatus('Undid Clear suggestions: ' + n + ' suggestion(s) are back.');
    }});
    setProposalStatus('Removed ' + n + ' unaccepted suggestion(s); accepted boxes are kept. Ctrl+Z undoes.');
}

// ── drawing ──────────────────────────────────────────────────────────────────

// Suggestions are dashed (orange = level 1, cyan = level 2); the selected one is
// thicker. Called from redraw() before the accepted boxes, which are solid.
function drawProposals() {
    var list = g_proposals[g_currentFrameNr];
    if (!list) return;
    g_context.save();
    for (var i = 0; i < list.length; i++) {
        var p = list[i];
        var selected = p === g_selectedProposal;
        g_context.beginPath();
        g_context.setLineDash([8, 6]);
        g_context.lineWidth = selected ? 4 : 2;
        g_context.strokeStyle = p.color;
        g_context.rect(p.x, p.y, p.width, p.height);
        g_context.stroke();
        g_context.font = (selected ? 'bold ' : '') + 'italic 13px sans-serif';
        g_context.fillStyle = p.color;
        g_context.fillText(p.label + '?', p.x + 4, p.y + p.height - 6);
    }
    g_context.restore();
}

// ── setup ────────────────────────────────────────────────────────────────────

function setupProposals() {
    updateProposalSettingsVisibility();
    setSelectedProposalControlsEnabled(false);
    $('#propLevels, #propTrack').change(updateProposalSettingsVisibility);

    $('#proposeButton').click(function() { requestProposals(g_currentFrameNr); });
    $('#clearProposalsButton').click(clearProposals);

    $('#renameTrackButton').click(renameSelectedTrack);
    $('#acceptTrackButton').click(acceptSelectedTrack);
    $('#rejectTrackButton').click(rejectSelectedTrack);

    // Ctrl+Enter accepts the selected suggestion (also from the name field);
    // plain Enter in the name field accepts the whole track (Accept track).
    $(document).keydown(function(e) {
        if (!g_selectedProposal) return;
        var inName = e.target.id === 'proposalName';
        if (e.which === 13 && e.ctrlKey) {
            e.preventDefault();
            acceptSelectedProposal();
        } else if (e.which === 13 && inName) {
            e.preventDefault();
            acceptSelectedTrack();
        } else if (e.which === 27 && inName) {
            selectProposal(null);
        }
    });
}
