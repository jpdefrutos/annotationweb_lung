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
var g_acceptKeyFrames = {};    // frame_nr -> true if accepting a suggestion made it a key frame
var g_coveredProposals = {};   // frame_nr -> suggestions of this load hidden under a box (shown again when the box goes)
var g_proposalsVisible = true;  // Show suggestions switch; hiding never deletes them
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

// Suggestions for all frames the task shows are loaded in the background when
// the page opens, chunk by chunk from the first frame, so the tracks continue
// over the whole video. They appear as each chunk arrives. The server keeps
// the results, so opening the video again is fast. A new load (Reload
// suggestions) replaces the unaccepted suggestions and stops an older load.
var PROPOSAL_CHUNK = 100;
var g_proposalLoad = 0;         // id of the current load; older responses are ignored

// Pressing Reload suggestions without changing a setting computes everything
// again instead of using the server's stored results (e.g. after a frame that
// could not be read has been fixed).
var g_lastProposalParams = null;

function loadAllProposals(fromButton) {
    if (g_imageSequenceID === null) return;
    var load = ++g_proposalLoad;
    var first = g_startFrame, last = g_startFrame + g_sequenceLength;
    var totals = {rows: 0, auto: 0, autoTracks: 0, dropped: 0, hidden: 0, skipped: []};
    g_proposals = {};
    g_coveredProposals = {};
    g_proposalState = null;
    if (g_selectedProposal) selectProposal(null);
    var params = JSON.stringify(proposalParams());
    var refresh = fromButton === true && params === g_lastProposalParams;
    g_lastProposalParams = params;
    var started = Date.now();

    function step(start) {
        if (load !== g_proposalLoad) return;
        var count = Math.min(PROPOSAL_CHUNK, last - start + 1);
        setProposalStatus('Finding suggestions for the whole video: frames ' + first + '–' +
            (start - 1) + ' of ' + first + '–' + last + ' done (' +
            Math.round(100 * (start - first) / (last - first + 1)) + '%)…' +
            (start === first ? ' The first time a video is opened this takes a few minutes.' : ''));
        $.ajax({
            type: 'POST',
            url: '/bronchoscopy-bounding-box/propose/',
            data: {
                task_id: g_taskID,
                image_sequence_id: g_imageSequenceID,
                start: start,
                count: count,
                params: params,
                state: g_proposalState ? JSON.stringify(g_proposalState) : '',
                refresh: refresh ? 1 : '',
            },
            dataType: 'json'
        }).done(function(data) {
            if (load !== g_proposalLoad) return;
            if (!data.success) { loadFailed(data.message || 'Suggestion failed'); return; }
            var r = storeProposalChunk(data);
            totals.rows += r.shown; totals.auto += r.auto.boxes; totals.autoTracks += r.auto.tracks;
            totals.dropped += data.dropped || 0; totals.hidden += r.hidden;
            totals.skipped = totals.skipped.concat(data.skipped || []);
            totals.skipMessage = totals.skipMessage || data.skip_message;
            if (r.auto.boxes)
                showProposalNotice(r.auto.boxes + ' box(es) continuing ' + r.auto.tracks +
                    ' accepted track(s) in frames ' + data.frames[0] + '–' + data.frames[1] +
                    ' were accepted automatically. Undoing that Accept track (Ctrl+Z) takes them back too.', true);
            if (g_selectedProposal) showSelectedProposalInfo();
            try { redrawSequence(); } catch (e) {}    // e.g. frame image not loaded: keep loading
            if (data.frames[1] < last) step(data.frames[1] + 1);
            else loadDone();
        }).fail(function(xhr) {
            if (load !== g_proposalLoad) return;
            loadFailed(xhr.responseJSON && xhr.responseJSON.message
                ? xhr.responseJSON.message : 'Suggestion failed (' + xhr.status + ')');
        });
    }

    function loadFailed(msg) {
        setProposalStatus(msg + ' Press Reload suggestions to try again.', true);
    }

    function loadDone() {
        setProposalStatus(totals.rows + ' suggestion' + (totals.rows === 1 ? '' : 's') +
            ' in frames ' + first + '–' + last + '.' +
            (totals.auto ? ' ' + totals.auto + ' box(es) continuing accepted tracks were accepted automatically.' : '') +
            (totals.dropped ? ' ' + totals.dropped + ' box(es) on short tracks were left out.' : '') +
            (totals.hidden ? ' ' + totals.hidden + ' hidden because they overlap a box you already have.' : '') +
            (totals.skipped.length ? ' Frame(s) ' + skippedText(totals.skipped) +
                ' could not be read and have no suggestions; tracks start again after them (' +
                totals.skipMessage + '). Once fixed, press Reload suggestions.' : ''),
            totals.skipped.length > 0);
    }

    function skippedText(list) {
        return list.length > 10 ? list.slice(0, 10).join(', ') + ' and ' + (list.length - 10) + ' more'
                                : list.join(', ');
    }

    step(first);
}

// Stores one chunk. When the chunk continues the previous one (the normal
// case), tracks accepted just before it go on being accepted (see
// autoAcceptContinuedTracks).
function storeProposalChunk(data) {
    var first = data.frames[0], last = data.frames[1];
    var continued = !!g_proposalState && data.state.run_id === g_proposalState.run_id &&
        g_proposalState.prev !== null && first === g_proposalState.prev + 1;
    g_proposalState = data.state;
    var hidden = storeProposals(data.rows, first, last);
    var auto = continued ? autoAcceptContinuedTracks(first, last) : {boxes: 0, tracks: 0};
    return {shown: data.rows.length - hidden - auto.boxes, hidden: hidden, auto: auto};
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
            if (list[i].tracked && list[i].label !== list[i].placeholder)
                consider(list[i], fn, list[i].label);
    }
    for (var g in g_boxes) {
        var gn = parseInt(g, 10);
        if (gn >= firstFrame) continue;
        var boxes = g_boxes[g];
        for (var j = 0; j < boxes.length; j++)
            if (boxes[j].src && boxes[j].src.tracked) consider(boxes[j].src, gn, boxes[j].label);
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
// The box that a suggestion lies on (so it is not shown), or null.
function coveredByBox(p) {
    var boxes = g_boxes[p.frame_nr] || [];
    for (var i = 0; i < boxes.length; i++)
        if (iou(p, boxes[i]) >= g_overlapHideIou) return boxes[i];
    return null;
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

// taken(letter), if given, says a letter is already used next to this one;
// a new track then skips it.
function letterFor(counterKey, trackKey, taken) {
    if (!(trackKey in g_trackLetters)) {
        var n = g_letterCounters[counterKey] || 0;
        while (taken && taken(letterName(n))) n++;
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
    // tracked and frame_nr too: without tracking, IDs only hold within a frame (sameTrack)
    var parent = { run_id: p.run_id, box_id: p.parent_id, tracked: p.tracked, frame_nr: p.frame_nr };
    var list = g_proposals[p.frame_nr] || [];
    for (var i = 0; i < list.length; i++)
        if (sameTrack(list[i], parent)) return list[i].label;
    var boxes = g_boxes[p.frame_nr] || [];
    for (var j = 0; j < boxes.length; j++) {
        var src = boxes[j].src || boxes[j].coveredSrc;
        if (src && sameTrack(src, parent)) return boxes[j].label;
    }
    return null;
}

// Sets r.letter and returns the suggested name. parentLabels holds the names
// of this response's main boxes ("frame|box_id" -> name), which may be hidden
// and so not findable through parentName(). frameRows (optional) are all rows
// of r's frame in this response.
function suggestedName(r, parentLabels, ownName, frameRows) {
    var tracked = r.label.charAt(0) === 'T';
    // A handed-over box normally takes its track's earlier name (see
    // syncHandedOverNames), so it only gets a letter of its own if it has none.
    if (r.was_nested && r.level === 1 && !ownName) return r.label;
    var scope = r.run_id + (tracked ? '' : '|f' + r.frame_nr);
    var trackKey = scope + '|' + r.box_id;
    if (r.level === 1) {
        var branch = frameBranch(r.frame_nr);
        if (!branch) return r.label;
        // skip letters other main suggestions on this frame already have, or
        // a box has (e.g. "1,2,1,a" saved in an earlier session)
        var takenMain = function(letter) {
            var rows = frameRows || [];
            for (var i = 0; i < rows.length; i++) {
                var s = rows[i];
                if (s !== r && s.level === 1 &&
                        g_trackLetters[branch + '|' + scope + '|' + s.box_id] === letter) return true;
            }
            return labelExistsInFrame(r.frame_nr, branch + ',' + letter);
        };
        r.letter = letterFor(scope + '|branch|' + branch, branch + '|' + trackKey, takenMain);
        return branch + ',' + r.letter;
    }
    var parent = parentLabels[r.frame_nr + '|' + r.parent_id] || parentName(r) || ('T' + r.parent_id);
    // Letters count per parent track. If the parent's track broke and goes on
    // under a new ID while its nested boxes kept theirs (and their letters),
    // the new parent's count starts at "a" again, so a new nested box skips
    // letters its siblings on this frame already have, or an accepted box has.
    var taken = function(letter) {
        var rows = frameRows || [];
        for (var i = 0; i < rows.length; i++) {
            var s = rows[i];
            if (s !== r && s.level === 2 && s.parent_id === r.parent_id &&
                    g_trackLetters[scope + '|' + s.box_id] === letter) return true;
        }
        return labelExistsInFrame(r.frame_nr, parent + ',' + letter);
    };
    r.letter = letterFor(scope + '|parent|' + r.parent_id, trackKey, taken);
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
            // still unrenamed: the suggested name, or the name taken from the
            // track (syncedLabel), which follows when the track is renamed
            if (p.was_nested && p.level === 1 &&
                    (p.label === p.placeholder || p.label === p.syncedLabel)) {
                if (last[key]) {
                    p.label = p.syncedLabel = last[key].label;
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
    var parentLabels = {}, byFrame = {};
    for (var f = firstFrame; f <= lastFrame; f++) {
        delete g_proposals[f];
        delete g_coveredProposals[f];
    }
    for (var j = 0; j < rows.length; j++)
        (byFrame[rows[j].frame_nr] = byFrame[rows[j].frame_nr] || []).push(rows[j]);
    for (var i = 0; i < rows.length; i++) {
        var r = rows[i];
        var tracked = r.tracked = r.label.charAt(0) === 'T';
        r.placeholder = suggestedName(r, parentLabels, false, byFrame[r.frame_nr]);
        r.label = r.placeholder;
        var key = r.run_id + '|' + r.box_id;
        if (tracked && key in names) r.label = names[key];
        r.load = g_proposalLoad;
        var cover = coveredByBox(r);
        if (r.level === 1) {
            // A main suggestion on an accepted box (e.g. saved in an earlier
            // session, so not linked to its suggestion) names its nested
            // suggestions after that box, and keeps doing so after a rename.
            parentLabels[r.frame_nr + '|' + r.box_id] = cover ? cover.label : r.label;
            if (cover && !cover.src)
                cover.coveredSrc = { run_id: r.run_id, box_id: r.box_id, tracked: !!r.tracked, frame_nr: r.frame_nr };
        }
        if (cover) {
            hidden++;
            (g_coveredProposals[r.frame_nr] = g_coveredProposals[r.frame_nr] || []).push(r);
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

// Undo puts a suggestion back, unless it is from an older load (before Reload
// suggestions): that load's suggestions were replaced, so it would show twice.
// For a box accepted before the Reload, the new load's suggestion for that
// opening was hidden under the box (g_coveredProposals); that one comes back.
function restoreProposal(p) {
    var f = p.frame_nr;
    p.acceptedBox = null;
    if (p.load === g_proposalLoad) {
        var c = (g_coveredProposals[f] || []).indexOf(p);
        if (c !== -1) g_coveredProposals[f].splice(c, 1);
        (g_proposals[f] = g_proposals[f] || []).push(p);
        return;
    }
    var covered = g_coveredProposals[f] || [];
    for (var i = 0; i < covered.length; i++) {
        var q = covered[i];
        if (iou(p, q) >= g_overlapHideIou && !coveredByBox(q)) {
            covered.splice(i, 1);
            (g_proposals[f] = g_proposals[f] || []).push(q);
            return;
        }
    }
}

// After boxes are deleted, moved, drawn or restored: a suggestion under a box
// is hidden (as when loading), and one whose box is gone is shown again.
function syncCoveredProposals() {
    var frames = {};
    for (var f in g_proposals) frames[f] = true;
    for (var g in g_coveredProposals) frames[g] = true;
    for (var fr in frames) {
        var shown = [], hidden = [];
        (g_proposals[fr] || []).concat(g_coveredProposals[fr] || []).forEach(function(p) {
            // an accepted suggestion stays hidden while its own box exists,
            // even if that box was resized or moved away from it
            var ownBox = p.acceptedBox && (g_boxes[fr] || []).indexOf(p.acceptedBox) !== -1;
            (ownBox || coveredByBox(p) ? hidden : shown).push(p);
        });
        if (shown.length) g_proposals[fr] = shown; else delete g_proposals[fr];
        if (hidden.length) g_coveredProposals[fr] = hidden; else delete g_coveredProposals[fr];
    }
    if (g_selectedProposal && !proposalExists(g_selectedProposal))
        selectProposal(null);
}

function proposalExists(p) {
    var list = g_proposals[p.frame_nr];
    return !!list && list.indexOf(p) !== -1;
}

// Without tracking ("D" boxes) the IDs are only numbers within one frame, so
// such a "track" is just the box itself.
function sameTrack(a, b) {
    return a.run_id === b.run_id && a.box_id === b.box_id &&
        (a.tracked || a.frame_nr === b.frame_nr);
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
    if (!g_proposalsVisible) return null;
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
    showSelectedProposalInfo();

    // Pre-fill the current name. For a suggested name, only its last part (the
    // letter) is selected, so typing "1" turns "1,2,1,a" into "1,2,1,1".
    var input = $('#proposalName');
    var el = input.val(p.label)[0];
    // no scrolling to the field: the image would move under the mouse
    el.focus({preventScroll: true});
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
    // addBox() marks the frame as a key frame; remember if that is new, so an
    // undo can take the mark away again (unacceptTrack)
    var newKeyFrame = g_targetFrames.indexOf(p.frame_nr) === -1;
    // no redraw per box: the callers redraw once when done
    var box = addBox(p.frame_nr, p.x, p.y, p.x + p.width, p.y + p.height, name, undefined, recordUndo, false, false);
    if (!box) return null;
    if (newKeyFrame) g_acceptKeyFrames[p.frame_nr] = true;
    // in memory only; for parentName() and continuing accepted tracks
    // frame_nr: without tracking, IDs only hold within a frame (sameTrack)
    box.src = { run_id: p.run_id, box_id: p.box_id, tracked: !!p.tracked, load: p.load, frame_nr: p.frame_nr };
    removeProposal(p);
    // kept under its box: if the box is deleted, the suggestion comes back
    p.acceptedBox = box;
    (g_coveredProposals[p.frame_nr] = g_coveredProposals[p.frame_nr] || []).push(p);
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
    var group = [{proposal: p, box: box}];
    box.acceptGroup = group;
    g_undoStack.push({type: 'custom', undo: function() {
        unacceptTrack(group, group.length === 1 ? 'Undid Accept: the box is a suggestion again.' : undefined);
    }});
    refreshChildNames();
    rebuildLabelDropdown();
    redrawSequence();
}

// Info for the selected suggestion: its whole track, and exactly which boxes
// the track buttons (Accept track / Enter) will take, so the annotator sees
// how long the track is before accepting it.
function showSelectedProposalInfo() {
    var p = g_selectedProposal;
    if (!p) return;
    var track = trackProposals(p);
    var scope = trackProposals(p, trackScopeFrame());
    var span = function(list) {
        return list[0].frame_nr === list[list.length - 1].frame_nr ? 'frame ' + list[0].frame_nr :
            'frames ' + list[0].frame_nr + '–' + list[list.length - 1].frame_nr;
    };
    var info = 'Suggestion <b>' + escapeHtml(p.label) + '</b>, ' +
        (p.level === 1 ? 'main box' : 'nested box') + '. Track: ' + track.length +
        ' frame' + (track.length === 1 ? '' : 's') + ' (' + span(track) + ').';
    if (scope.length && scope.length !== track.length)
        info += '<br>Accept track takes <b>' + scope.length + ' box' + (scope.length === 1 ? '' : 'es') +
            '</b> (' + span(scope) + ').';
    else if (scope.length === 1)
        info += '<br>Accept track takes only this box.';
    else if (scope.length)
        info += '<br>Accept track takes <b>all ' + scope.length + '</b>.';
    var gaps = track.length > 1 ? track[track.length - 1].frame_nr - track[0].frame_nr + 1 - track.length : 0;
    if (gaps > 0)
        info += ' Missing on ' + gaps + ' frame' + (gaps === 1 ? '' : 's') + ' in between.';
    $('#selectedProposalInfo').html(info);
}

// "From this frame onward" counts from the selected suggestion's frame (not
// the frame on screen), so the info line and the action always agree.
function trackScopeFrame() {
    var p = g_selectedProposal;
    return p && $('#proposalFromHere').is(':checked') ? p.frame_nr : undefined;
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
    g_undoStack.push({type: 'custom', boxes: false, undo: function() {
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
    accepted.track = true;      // Accept track: later chunks continue it (autoAcceptContinuedTracks)
    for (var i = 0; i < list.length; i++) {
        var box = acceptProposalAs(list[i], name, false);
        if (box) {
            accepted.push({proposal: list[i], box: box});
            box.acceptGroup = accepted;
        } else {
            skipped.push(list[i].frame_nr);
        }
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
    // max_lost as the server's tracker used it for this load (not the field,
    // which may have been changed since)
    var params = g_proposalState && g_proposalState.params;
    var lookback = (params && params.max_lost !== undefined ? params.max_lost
                    : parseInt($('#propMaxLost').val(), 10) || 0) + 1;
    // "run|box" -> {name, group}: tracks accepted from this load's suggestions
    // (after Reload, boxes from the earlier load do not count: the annotator
    // may have rejected or deleted parts of those tracks)
    var accepted = {};
    for (var f = first - lookback; f < first; f++) {
        var boxes = g_boxes[f] || [];
        for (var i = 0; i < boxes.length; i++) {
            var src = boxes[i].src;
            // only Accept track continues; a single accept (Ctrl+Enter) means only that box
            if (src && src.tracked && src.load === g_proposalLoad &&
                    boxes[i].acceptGroup && boxes[i].acceptGroup.track)
                accepted[src.run_id + '|' + src.box_id] = {name: boxes[i].label, group: boxes[i].acceptGroup};
        }
    }
    if (Object.keys(accepted).length === 0) return {boxes: 0, tracks: 0};
    var done = [], tracks = {};
    for (var g = first; g <= last; g++) {
        var list = (g_proposals[g] || []).slice();
        list.sort(function(a, b) { return a.level - b.level; });   // parents first
        for (var k = 0; k < list.length; k++) {
            var key = list[k].run_id + '|' + list[k].box_id;
            if (!list[k].tracked || !(key in accepted)) continue;
            var box = acceptProposalAs(list[k], accepted[key].name, false);
            if (box) {
                // Joins the Accept (track) it continues: undoing that accept
                // takes these back too. No undo step of its own, so a chunk
                // arriving in the background never comes between the
                // annotator's own actions on the Ctrl+Z list.
                var entry = {proposal: list[k], box: box};
                accepted[key].group.push(entry);
                box.acceptGroup = accepted[key].group;
                done.push(entry);
                tracks[key] = true;
            }
        }
    }
    if (done.length > 0) {
        refreshChildNames();
        rebuildLabelDropdown();
    }
    return {boxes: done.length, tracks: Object.keys(tracks).length};
}

// Undo for Accept / Accept track: removes the accepted boxes and turns them
// back into suggestions.
function unacceptTrack(accepted, notice) {
    // A frame that an accept made a key frame (whichever accept, e.g. a parent
    // and its child on the same frame undone in any order) loses the mark
    // again once it is empty, else saving would store an empty annotated frame.
    var touched = {};
    for (var i = 0; i < accepted.length; i++) {
        var a = accepted[i], f = a.proposal.frame_nr;
        var boxes = g_boxes[f] || [];
        var k = boxes.indexOf(a.box);
        if (k !== -1) boxes.splice(k, 1);
        touched[f] = true;
        restoreProposal(a.proposal);
    }
    for (var m in touched) {
        var fn = parseInt(m, 10);
        if (g_acceptKeyFrames[fn] && !(g_boxes[fn] && g_boxes[fn].length)) {
            removeKeyFrame(fn);
            delete g_acceptKeyFrames[fn];
        }
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
    g_undoStack.push({type: 'custom', boxes: false, undo: function() {
        for (var j = 0; j < list.length; j++) {
            restoreProposal(list[j]);
        }
        refreshChildNames();
        showProposalNotice('Undid Reject track: ' + list.length + ' suggestion(s) are back.');
    }});
    redrawSequence();
    showProposalNotice('Rejected ' + describeRange(list) + '. Ctrl+Z undoes.');
}

function rejectProposal(p) {
    removeProposal(p);
    g_undoStack.push({type: 'custom', boxes: false, undo: function() {
        restoreProposal(p);
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

// Show suggestions switch: hides (never deletes) all unaccepted suggestions,
// e.g. in parts of the video where copying boxes is easier and the dashed
// boxes would hide the openings. Accepted boxes are always shown.
function setProposalsVisible(visible) {
    g_proposalsVisible = visible;
    if (!visible) {
        selectProposal(null);
        hideProposalHover();
    }
    redrawSequence();
}

// ── hover ────────────────────────────────────────────────────────────────────

// Hovering a suggestion shows how many frames from here on belong to the same
// track, without having to click it.
var g_hover = null;             // {proposal, html, time} of the hovered suggestion

function updateProposalHover(e, x, y) {
    var p = proposalUnderCursor(x, y, isInsideBox(x, y));
    if (!p) { hideProposalHover(); return; }
    // hoverText scans the whole video's suggestions, so it is only redone when
    // another suggestion is hovered, or after half a second (rename, accept...)
    var now = Date.now();
    if (!g_hover || g_hover.proposal !== p || now - g_hover.time > 500)
        g_hover = {proposal: p, html: hoverText(p), time: now};
    var tip = $('#proposalHover');
    if (tip.length === 0)
        tip = $('<div id="proposalHover"></div>').appendTo('body');
    tip.html(g_hover.html).css({left: e.pageX + 14, top: e.pageY + 14}).show();
}

function hideProposalHover() {
    g_hover = null;
    $('#proposalHover').hide();
}

function hoverText(p) {
    var ahead = trackProposals(p, p.frame_nr), track = trackProposals(p);
    var last = ahead[ahead.length - 1].frame_nr;
    var gaps = last - p.frame_nr + 1 - ahead.length;
    var html = '<b>' + escapeHtml(p.label) + '?</b><br>';
    if (ahead.length === 1)
        html += 'Same track: only this frame';
    else
        html += 'Same track: <b>' + ahead.length + ' frames</b> from here (to frame ' + last + ')' +
            (gaps > 0 ? ', missing on ' + gaps : '');
    if (track[0].frame_nr < p.frame_nr)
        html += '<br>Whole track: frames ' + track[0].frame_nr + '–' + track[track.length - 1].frame_nr;
    return html;
}

// ── drawing ──────────────────────────────────────────────────────────────────

// Suggestions are dashed (orange = level 1, cyan = level 2); the selected one is
// thicker. Called from redraw() before the accepted boxes, which are solid.
function drawProposals() {
    if (!g_proposalsVisible) return;
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
    $('#proposalFromHere').change(showSelectedProposalInfo);
    $('#propLevels, #propTrack, #proposalPanel .proposal-settings input').change(function() {
        if (this.id !== 'showProposals')
            setProposalStatus('Press Reload suggestions to use the new settings.');
    });

    // a key may change the frame or the suggestion under the cursor; the hover
    // box comes back with the next mouse move
    $(document).keydown(hideProposalHover);
    $('#reloadProposalsButton').click(function() { loadAllProposals(true); });
    $('#showProposals').change(function() { setProposalsVisible(this.checked); });

    $('#renameTrackButton').click(renameSelectedTrack);
    $('#acceptTrackButton').click(acceptSelectedTrack);
    $('#rejectTrackButton').click(rejectSelectedTrack);

    // Ctrl+Enter accepts the selected suggestion (also from the name field);
    // plain Enter in the name field accepts the whole track (Accept track).
    $(document).keydown(function(e) {
        if (!g_selectedProposal) return;
        var inName = e.target.id === 'proposalName';
        if (e.which === 13 && e.ctrlKey) {
            if (!inName && isTextEntry(e.target)) return;  // e.g. typing a comment
            e.preventDefault();
            acceptSelectedProposal();
        } else if (e.which === 13 && inName) {
            e.preventDefault();
            acceptSelectedTrack();
        } else if (e.which === 27 && inName) {
            selectProposal(null);
        }
    });

    loadAllProposals();
}
