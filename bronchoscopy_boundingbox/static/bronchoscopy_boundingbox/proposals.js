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
    var count = parseInt($('#propCount').val(), 10) || 30;
    g_proposalBusy = true;
    $('#proposeButton, #proposeNextButton').prop('disabled', true);
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
        g_proposalState = data.state;
        storeProposals(data.rows, data.frames[0], data.frames[1]);
        var n = data.rows.length;
        setProposalStatus(n + ' suggestion' + (n === 1 ? '' : 's') + ' in frames ' +
            data.frames[0] + '–' + data.frames[1] +
            (continued ? ' (tracks continued)' : ' (new run ' + data.state.run_id + ')') + '.');
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
        $('#proposeNextButton').prop('disabled', !canContinueProposals());
    });
}

function canContinueProposals() {
    return g_proposalState !== null && g_proposalState.prev !== null &&
        g_proposalState.prev + 1 < g_startFrame + g_sequenceLength;
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

// New suggestions for a frame replace any older, unaccepted ones on it, so
// two runs never show overlapping placeholder names on the same frame. A
// suggestion whose track was already named in an earlier frame (e.g. in the
// previous chunk) gets that name, so a track is only named once.
function storeProposals(rows, firstFrame, lastFrame) {
    var names = earlierTrackNames(firstFrame);
    for (var f = firstFrame; f <= lastFrame; f++)
        delete g_proposals[f];
    for (var i = 0; i < rows.length; i++) {
        var r = rows[i];
        r.placeholder = r.label;
        var key = r.run_id + '|' + r.box_id;
        if (key in names) r.label = names[key];
        if (!(r.frame_nr in g_proposals))
            g_proposals[r.frame_nr] = [];
        g_proposals[r.frame_nr].push(r);
    }
    if (g_selectedProposal && !proposalExists(g_selectedProposal))
        selectProposal(null);
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

// Current name of a level-2 proposal's parent: the parent suggestion on the
// same frame, or the accepted box made from it. Linked by (run_id, parent_id),
// never by name, so renaming cannot break the link.
function parentName(p) {
    if (p.level !== 2 || p.parent_id === null) return null;
    var parent = { run_id: p.run_id, box_id: p.parent_id };
    var list = g_proposals[p.frame_nr] || [];
    for (var i = 0; i < list.length; i++)
        if (sameTrack(list[i], parent))
            return list[i].label !== list[i].placeholder ? list[i].label : null;
    var boxes = g_boxes[p.frame_nr] || [];
    for (var j = 0; j < boxes.length; j++)
        if (boxes[j].src && sameTrack(boxes[j].src, parent))
            return boxes[j].label;
    return null;
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

    // Pre-fill: the name given so far; for an unnamed nested box, its parent's
    // name, so e.g. "RB1" only needs a letter added to become "RB1a".
    var name = p.label;
    if (p.label === p.placeholder) {
        var pn = parentName(p);
        if (pn) name = pn;
    }
    $('#proposalName').val(name);
    $('#proposalName').focus().select();
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

// Turns a suggestion into an ordinary box with the given name. Returns false if
// the frame already has a box with that name (or the box is too small).
function acceptProposalAs(p, name) {
    var box = addBox(p.frame_nr, p.x, p.y, p.x + p.width, p.y + p.height, name, undefined, true, false);
    if (!box) return false;
    box.src = { run_id: p.run_id, box_id: p.box_id };   // in memory only; for parentName()
    removeProposal(p);
    g_annotationHasChanged = true;
    return true;
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
    if (!acceptProposalAs(p, name)) {
        alert('This frame already has a box named "' + name + '" (or the box is too small).');
        return;
    }
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
    if (list.length > 1 && !confirm('Rename ' + describeRange(list) + ' to "' + name + '"?')) return;
    for (var i = 0; i < list.length; i++)
        list[i].label = name;
    selectProposal(p);
}

function acceptSelectedTrack() {
    var p = g_selectedProposal;
    if (!p) return;
    var name = selectedName();
    if (!name) { alert('Enter a name first.'); return; }
    if (!confirmPlaceholder(name, p)) return;
    var list = trackProposals(p, trackScopeFrame());
    if (list.length === 0) return;
    if (list.length > 1 && !confirm('Accept ' + describeRange(list) + ' as "' + name + '"?')) return;
    var skipped = [];
    for (var i = 0; i < list.length; i++) {
        if (!acceptProposalAs(list[i], name))
            skipped.push(list[i].frame_nr);
    }
    rebuildLabelDropdown();
    redrawSequence();
    if (skipped.length > 0)
        alert('Not accepted on frame(s) ' + skipped.join(', ') + ': a box named "' + name +
            '" already exists there (or the box is too small). Those suggestions are kept.');
}

function rejectSelectedTrack() {
    var p = g_selectedProposal;
    if (!p) return;
    var list = trackProposals(p, trackScopeFrame());
    if (list.length > 1 && !confirm('Reject ' + describeRange(list) + '?')) return;
    for (var i = 0; i < list.length; i++)
        removeProposal(list[i]);
    redrawSequence();
}

function rejectProposalAt(x, y) {
    var p = proposalAt(x, y);
    if (!p) return;
    removeProposal(p);
    redrawSequence();
}

function acceptAllProposalsInFrame() {
    var list = (g_proposals[g_currentFrameNr] || []).slice();
    if (list.length === 0) return;
    var unnamed = list.filter(function(p) { return p.label === p.placeholder; }).length;
    if (unnamed > 0 && !confirm(unnamed + ' of ' + list.length +
            ' suggestion(s) on this frame still have temporary names. Accept all anyway?'))
        return;
    var skipped = [];
    // Parents first, so they are named before children that look them up.
    list.sort(function(a, b) { return a.level - b.level; });
    for (var i = 0; i < list.length; i++) {
        if (!acceptProposalAs(list[i], list[i].label))
            skipped.push(list[i].label);
    }
    rebuildLabelDropdown();
    redrawSequence();
    if (skipped.length > 0)
        alert('Not accepted (name already used on this frame, or box too small): ' + skipped.join(', '));
}

function clearProposals() {
    var n = 0;
    for (var f in g_proposals) n += g_proposals[f].length;
    if (n === 0) return;
    if (!confirm('Remove all ' + n + ' unaccepted suggestion(s)? Accepted boxes are kept.')) return;
    g_proposals = {};
    g_proposalState = null;     // the next run starts fresh
    $('#proposeNextButton').prop('disabled', true);
    selectProposal(null);
    setProposalStatus('');
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
    $('#proposeNextButton').click(function() {
        if (canContinueProposals()) requestProposals(g_proposalState.prev + 1);
    });
    $('#acceptFrameProposalsButton').click(acceptAllProposalsInFrame);
    $('#clearProposalsButton').click(clearProposals);

    $('#acceptProposalButton').click(acceptSelectedProposal);
    $('#rejectProposalButton').click(function() {
        if (g_selectedProposal) { removeProposal(g_selectedProposal); redrawSequence(); }
    });
    $('#renameTrackButton').click(renameSelectedTrack);
    $('#acceptTrackButton').click(acceptSelectedTrack);
    $('#rejectTrackButton').click(rejectSelectedTrack);

    // Ctrl+Enter accepts the selected suggestion, also from the name field;
    // plain Enter in the name field does the same.
    $(document).keydown(function(e) {
        if (!g_selectedProposal) return;
        var inName = e.target.id === 'proposalName';
        if (e.which === 13 && (e.ctrlKey || inName)) {
            e.preventDefault();
            acceptSelectedProposal();
        } else if (e.which === 27 && inName) {
            selectProposal(null);
        }
    });
}
