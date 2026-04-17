let g_backgroundImage;
let g_frameNr;
let g_currentColor = null;
let g_labels = {}; // Dictionary with keys frame_nr which each has a label
let g_labelColorMap = {}; // Maps label name -> hex color, to ensure unique colors

let g_selectedLabels = [];
let g_subsequenceStartFrame = null; // Start frame of subsequence
// Store custom labels for each frame
let g_customFrameLabels = {};
let g_subsequenceCustomLabel = null;


function isTextboxOnlyMode() {
    // Activate textbox mode if any label is named "textbox"
    return g_labelButtons.some(
        l => typeof l.name === 'string' && l.name.toLowerCase() === 'textbox'
    );
}

function isSingleNonTextboxMode() {
    return g_labelButtons.length === 1 && !isTextboxOnlyMode();
}

function loadSubsequenceClassificationTask() {
    console.log('In load subsequence classification');

    // Only set up multi\-label button toggling if there are >1 labels
    if (g_labelButtons.length > 1) {
        g_selectedLabels = [];
        for (let i = 0; i < g_labelButtons.length; ++i) {
            let label_id = g_labelButtons[i].id;
            $('#labelButton' + label_id).click(function () {
                $(this).toggleClass('activeLabel');
                if ($(this).hasClass('activeLabel')) {
                    if (!g_selectedLabels.includes(label_id)) {
                        g_selectedLabels.push(label_id);
                    }
                } else {
                    g_selectedLabels = g_selectedLabels.filter(id => id !== label_id);
                }
                console.log('Selected labels:', g_selectedLabels);
            });
        }
    }

    setupSubsequenceClassification();

    const startSubsequence = document.querySelector('.startSubsequenceButton');
    const endSubsequence = document.querySelector('.endSubsequenceButton');
    startSubsequence.onclick = startButtonClick;
    endSubsequence.onclick = endButtonClick;
}

// helper to convert "#rrggbb" to {red, green, blue}
function hexToRgbObject(hex) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!m) return null;
    return {
        red: parseInt(m[1], 16),
        green: parseInt(m[2], 16),
        blue: parseInt(m[3], 16),
    };
}

// Build a dict: label name -> {red, green, blue}
function buildCustomLabelColors() {
    const colors = {};
    for (const [frameStr, labelName] of Object.entries(g_customFrameLabels)) {
        const hex = stringToColor(labelName);
        const rgb = hexToRgbObject(hex);
        if (rgb && !colors[labelName]) {
            colors[labelName] = rgb;
        }
    }
    return colors;
}


// Function to send data for saving
function sendDataForSave() {
    // build mapping: label name -> {red, green, blue}
    const customLabelColors = buildCustomLabelColors();

    return $.ajax({
        type: "POST",
        url: "/subsequence-classification/save/",
        data: {
            image_id: g_imageID,
            task_id: g_taskID,
            frame_labels: JSON.stringify(g_labels),
            custom_frame_labels: JSON.stringify(g_customFrameLabels), // Send custom labels as JSON string
            custom_frame_label_colors: JSON.stringify(customLabelColors),
            target_frames: JSON.stringify(g_targetFrames),
            quality: $('input[name=quality]:checked').val() || 'unknown', // Default to unknown if no quality is selected
            rejected: g_rejected ? 'true':'false',
            comments: $('#comments').val(),
        },
        dataType: "json" // Need this do get result back as JSON
    });
}

function updateCurrentFrameLabelDisplay(label) {
    const leftLabel = document.getElementById('currentFrameLabel');
    const rightLabel = document.getElementById('currentFrameLabelDisplay');
    if (leftLabel) leftLabel.textContent = label;
    if (rightLabel) rightLabel.textContent = label;
}

// javascript
function startButtonClick(e) {
    // 1) Textbox\-only mode
    if (isTextboxOnlyMode()) {
        const inputEl = document.getElementById('customLabelInput');
        if (!inputEl) {
            alert('Textbox element not found. Please check the template for the textbox only configuration.');
            return;
        }
        const label = inputEl.value.trim();
        if (!label) {
            alert('Please enter a label before marking a subsequence!');
            return;
        }
        g_subsequenceStartFrame = g_currentFrameNr;
        g_subsequenceCustomLabel = label;
        g_subsequenceLabels = null;
        updateCurrentFrameLabelDisplay(label);
        console.log('Start subsequence (textbox) at frame', g_currentFrameNr, 'Label:', label);
        return;
    }

    // 2) Multi\-label mode: two or more labels
    if (g_labelButtons.length > 1 &&
    !g_labelButtons.some(l => typeof l.name === 'string' && l.name.toLowerCase() === 'textbox')
    ) {
        if (!Array.isArray(g_selectedLabels) || g_selectedLabels.length === 0) {
            alert('You need to select at least one label before marking a subsequence!');
            return;
        }
        g_subsequenceStartFrame = g_currentFrameNr;
        g_subsequenceCustomLabel = null;
        g_subsequenceLabels = [...g_selectedLabels];

        const labelNames = g_subsequenceLabels
            .map(id => getLabelWithId(id))
            .filter(Boolean)
            .map(l => l.name)
            .join(', ');

        updateCurrentFrameLabelDisplay(labelNames);
        console.log('Start subsequence (multi label) at frame', g_currentFrameNr, 'Labels:', g_subsequenceLabels);
        return;
    }

    // 3) Single non\-textbox label
    if (isSingleNonTextboxMode()) {
        g_subsequenceStartFrame = g_currentFrameNr;
        g_subsequenceCustomLabel = null;
        g_subsequenceLabels = [g_labelButtons[0].id];

        updateCurrentFrameLabelDisplay(g_labelButtons[0].name);
        console.log('Start subsequence (single label) at frame', g_currentFrameNr, 'Label:', g_subsequenceLabels[0]);
        return;
    }

    alert('Configuration error: no valid subsequence mode detected.');
}

function endButtonClick(e) {
    // 1) Textbox\-only mode
    if (isTextboxOnlyMode()) {
        if (g_subsequenceStartFrame === null || g_subsequenceCustomLabel === null) {
            alert('You need to start a subsequence first!');
            return;
        }

        //const color = '#ff0000';
        const color = stringToColor(g_subsequenceCustomLabel); // Generate color from label text

        for (let frame = g_subsequenceStartFrame; frame <= g_currentFrameNr; frame++) {
            g_customFrameLabels[frame] = g_subsequenceCustomLabel;
            setupSliderMark(frame, color);
        }

        const nextFrame = g_currentFrameNr + 1;
        if (nextFrame < g_framesLoaded) {
            g_customFrameLabels[nextFrame] = g_subsequenceCustomLabel;
            setupSliderMark(nextFrame, color);
        }

        g_subsequenceStartFrame = null;
        g_subsequenceCustomLabel = null;
        updateFrameLabelVariables();
        return;
    }

    // 2) Multi\-label mode (two or more labels)
    if (g_labelButtons.length > 1) {
        if (g_subsequenceStartFrame === null ||
            !Array.isArray(g_subsequenceLabels) ||
            g_subsequenceLabels.length === 0) {
            alert('You need to start a subsequence first!');
            return;
        }

        const start = Math.min(g_subsequenceStartFrame, g_currentFrameNr);
        const end = Math.max(g_subsequenceStartFrame, g_currentFrameNr);

        for (let frameIdx = start; frameIdx <= end; frameIdx++) {
            addKeyFrame(frameIdx);
            setLabel(frameIdx, g_subsequenceLabels);
        }

        sliderMarkSubsequence(start, end, getLabelWithId(g_subsequenceLabels[0]));
        g_subsequenceStartFrame = null;
        g_subsequenceLabels = [];
        updateFrameLabelVariables();
        return;
    }

    // 3) Single non\-textbox label
    if (isSingleNonTextboxMode()) {
        if (g_subsequenceStartFrame === null ||
            !Array.isArray(g_subsequenceLabels) ||
            g_subsequenceLabels.length === 0) {
            alert('You need to start a subsequence first!');
            return;
        }

        const start = Math.min(g_subsequenceStartFrame, g_currentFrameNr);
        const end = Math.max(g_subsequenceStartFrame, g_currentFrameNr);

        for (let frameIdx = start; frameIdx <= end; frameIdx++) {
            addKeyFrame(frameIdx);
            setLabel(frameIdx, g_subsequenceLabels);
        }

        sliderMarkSubsequence(start, end, getLabelWithId(g_subsequenceLabels[0]));
        g_subsequenceStartFrame = null;
        g_subsequenceLabels = [];
        updateFrameLabelVariables();
        return;
    }

    alert('Configuration error: no valid subsequence mode detected.');
}


function setupSubsequenceClassification() {
    console.log('Setting up subsequence classification....');

    // Define event callbacks
    $('#clearButton').click(function () {
        g_annotationHasChanged = true;
        console.log('Clearing labels');
        // Reset image quality form
        $('#imageQualityForm input[type="radio"]').each(function () {
            $(this).prop('checked', false);
        });
        // remove all labels
        for(var i = 0; i < g_labelButtons.length; i++)  {
            $('#labelButton' + g_labelButtons[i].id).removeClass('activeLabel');
        }
        g_labels = {}; // Remove all labels
        g_targetFrames = []; // Remove all target frames
        removeAllSliderMarks(); // Remove all slider marks
        g_customFrameLabels = {};
        g_subsequenceStartFrame = null;
        g_subsequenceCustomLabel = null;
         if (g_labelButtons.length > 1) {
             g_currentLabel = -1; // Set all label buttons to inactive
         }

    });

    //changeLabel(g_labelButtons[0].id);    // Set first label active
    redrawSequence();

    for(let frame_nr = 0; frame_nr < g_targetFrames.length; ++frame_nr) {
        if (frame_nr in g_labels) {
            let frameLabelId = g_labels[frame_nr];
            let label = g_labelButtons[frameLabelId];
            let labelColor = frameLabelId.color;
            console.log(frameLabelId, labelColor);

            addKeyFrame(frame_nr, labelColor);
            setLabel(frame_nr, frameLabelId);
        }
    }

}

function findNextFrameWithDifferentLabel(frameIdx, labelId) {
    let lastFrame = g_startFrame + g_sequenceLength;
    for (let i = frameIdx; i <= lastFrame; i++) {
        if (g_labels[i] !== labelId) {
            return i;
        }
    }
    return -1;
}

function findPreviousFrameWithSameLabel(frameIdx, labelId) {
    let lastFrame = g_startFrame + g_sequenceLength;
    for (let i = frameIdx; i >= g_startFrame; i--) {
        if (g_labels[i] === labelId) {
            return i;
        }
    }
    return -1;
}

/*
    Functions for updating and redrawing the subsequence labels
 */

function redrawSequence() {
    let index = g_currentFrameNr - g_startFrame;
    g_context.drawImage(g_sequence[index], 0, 0, g_canvasWidth, g_canvasHeight); // Draw background image
}

function updateSliderForLabelChange(frame_nr, label_ids) {
    // Ensure label_ids is always an array
    if (!Array.isArray(label_ids)) {
        label_ids = [label_ids];
    }

    // Check for custom label
    if (g_customFrameLabels[frame_nr]) {
        const customColor = stringToColor(g_customFrameLabels[frame_nr]);
        setupSliderMark(frame_nr, customColor);
    } else {
        // Use the first label's color
        const label = getLabelWithId(label_ids[0]);
        if (label) {
            const hexColor = colorToHexString(label.red, label.green, label.blue);
            setupSliderMark(frame_nr, hexColor);
        }
    }
}


function setLabel(frame_nr, label_ids) {
    // Ensure label_ids is always an array
    if (!Array.isArray(label_ids)) {
        label_ids = [label_ids];
    }
    g_labels[frame_nr] = label_ids;

    // Update slider marker for frame (use first label's color or customize)
    //let label = getLabelWithId(label_ids[0]);
    //let hexColor = colorToHexString(label.red, label.green, label.blue);
    //setupSliderMark(frame_nr, hexColor);

    // Update the slider marker
    updateSliderForLabelChange(frame_nr, label_ids);
}

function addSubsequenceLabel(frame_nr, label_id) {
    addKeyFrame(frame_nr);
    // Ensure g_labels[frame_nr] is an array
    if (!g_labels[frame_nr]) {
        g_labels[frame_nr] = [];
    }
    if (!g_labels[frame_nr].includes(label_id)) {
        g_labels[frame_nr].push(label_id);
    }
    // Label all frames in a subsequence
    //for (let frame = g_subsequenceStartFrame; frame <= g_subsequenceEndFrame; frame++) {
    // addSubsequenceLabel(frame, label_id);
    //}
    // Optionally update slider marker for the first label
    //let label = getLabelWithId(g_labels[frame_nr][0]);
    //if (label) {
    //    let hexColor = colorToHexString(label.red, label.green, label.blue);
    //    setupSliderMark(frame_nr, hexColor);
    //}

    // Update the slider marker
    updateSliderForLabelChange(frame_nr, g_labels[frame_nr]);

}

function stringToColor(str) {
    // Simple hash to color
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    let color = '#';
    for (let i = 0; i < 3; i++) {
        let value = (hash >> (i * 8)) & 0xFF;
        color += ('00' + value.toString(16)).substr(-2);
    }
    return color;
}


function updateFrameLabelVariables() {
    // --- Update current frame label --- // Check for custom label first
    const customLabel = g_customFrameLabels[g_currentFrameNr];
    if (customLabel) {
        const color = stringToColor(customLabel); // Generate color from label text
        $('#currentFrameLabel').text(customLabel);
        $('#currentFrameLabelDisplay').text(customLabel);

        $('#currentFrameLabel').html(`<span style="color: ${color}">${customLabel}</span>`);
        $('#currentFrameLabelDisplay').html(
            `<span style="color: ${color}">${customLabel}</span>`
        );
        return;
    }
    const labelIds = g_labels[g_currentFrameNr] || [];
    if (labelIds.length > 0) {
        //const labelObjs = labelIds.map(id => getLabelWithId(id));
        // Filter out nulls to avoid errors
        const labelObjs = labelIds.map(id => getLabelWithId(id)).filter(label => label !== null);
        const labelNames = labelObjs.map(label => label.name).join(', ');
        $('#currentFrameLabel').text(labelNames);

        const styledLabels = labelObjs.map(label => {
            return `<span style="color: ${colorToHexString(label.red, label.green, label.blue)}">${label.name}</span>`;
        }).join(', ');
        $('#currentFrameLabelDisplay').html(styledLabels);
    } else {
        $('#currentFrameLabel').text('No label');
        $('#currentFrameLabelDisplay').text('No label');
    }


    // --- Update predicted class or branch code ---
    console.log('Current frame:', g_currentFrameNr);
    console.log('Branch codes:', window.branchCodes[g_currentFrameNr]);
    console.log('Frame predictions:', window.framePredictions[g_currentFrameNr]);
    const predictedClassElem = document.getElementById('predicted-class-info');
    //const predictedClassHeader = predictedClassElem.parentElement; // the <h3> that contains the span
    const predictedClassHeader = predictedClassElem ? predictedClassElem.parentElement : undefined;

    if (predictedClassElem) {
        let value = '';
        let label = '';

        const usePred = (typeof window.usePredictions !== 'undefined')
            ? window.usePredictions
            : true;

        if (usePred) {
            if (window.framePredictions &&
                window.framePredictions[g_currentFrameNr] !== undefined) {
                value = window.framePredictions[g_currentFrameNr];
                label = 'Predicted class: ';
            }
        } else {
            if (window.branchCodes &&
                window.branchCodes[g_currentFrameNr] !== undefined) {
                value = window.branchCodes[g_currentFrameNr];
                label = 'Branch code: ';
            } else if (window.branchCodes) {
                value = 'Undefined';
                label = 'Branch code: ';
            }
        }
        predictedClassElem.textContent = value;
        predictedClassElem.parentElement.firstChild.textContent = label;
        predictedClassElem.parentElement.style.display = label ? 'block' : 'none';
    }

    console.log('Predicted class info:', window.framePredictions[g_currentFrameNr]);
}


function dictDelete(dict, key) {
    if (dict.hasOwnProperty(key)) {
        delete dict[key];
        return dict;
    }
    return false;
}

/*
    Overwrite functions from annotationweb.js
*/

// Overload loadSequence() function in annotationweb.js
function loadSequence(
    image_sequence_id, start_frame, nrOfFrames, show_entire_sequence,
    user_frame_selection, annotate_single_frame, frames_to_annotate,
    images_to_load_before, images_to_load_after, auto_play) {

    // If user cannot select frame, and there are no target frames, select last frame as target frame
    if(!user_frame_selection && annotate_single_frame && frames_to_annotate.length === 0) {
        // Select last frame as target frame
        frames_to_annotate.push(nrOfFrames-1);
    }
    g_userFrameSelection = user_frame_selection;


    console.log('In load sequence');
    // Create play/pause button
    setPlayButton(auto_play);
    $("#playButton").click(function() {
        setPlayButton(!g_isPlaying);
        if(g_isPlaying) // Start it again
            incrementFrame();
    });

    // Create canvas
    var canvas = document.getElementById('canvas');
    canvas.setAttribute('width', g_canvasWidth);
    canvas.setAttribute('height', g_canvasHeight);
    // IE stuff
    if(typeof G_vmlCanvasManager != 'undefined') {
        canvas = G_vmlCanvasManager.initElement(canvas);
    }
    g_context = canvas.getContext("2d");


    if(g_targetFrames.length > 0) {
        g_currentFrameNr = g_targetFrames[0];
    } else {
        g_currentFrameNr = 0;
    }
    $('#currentFrame').text(g_currentFrameNr);
    updateFrameLabelVariables();

    var start;
    var end;
    var totalToLoad;
    if(show_entire_sequence || !annotate_single_frame) {
        start = start_frame;
        end = start_frame + nrOfFrames - 1;
        totalToLoad = nrOfFrames;
    } else {
        start = max(start_frame, g_currentFrameNr - images_to_load_before);
        end = min(start_frame + nrOfFrames - 1, g_currentFrameNr + images_to_load_after);
        totalToLoad = end - start;
    }
    g_startFrame = start;
    g_sequenceLength = end-start;
    console.log("Start frame = " + g_startFrame.toString() + ", sequence length = " + g_sequenceLength.toString());

    // Create slider
    $("#slider").slider({
        range: "max",
        min: start,
        max: end,
        step: 1,
        value: g_currentFrameNr,
        create: function() {
            var handle = $(this).find('.ui-slider-handle');
            var width = $(this).width();
            handle.css({
                'width': width * 0.02,
                'margin-left': 0,
                'margin-right': 0
            });
        },
        slide: function(event, ui) {
            g_currentFrameNr = ui.value;
            $('#currentFrame').text(g_currentFrameNr);
            updateFrameLabelVariables();
            setPlayButton(false);
            redrawSequence();
        }
    });

    // Create progress bar
    g_progressbar = $( "#progressbar" );
    var progressLabel = $(".progress-label");
    g_progressbar.progressbar({
      value: false,
      change: function() {
        progressLabel.text( "Please wait while loading. " + g_progressbar.progressbar( "value" ).toFixed(1) + "%" );
      },
      complete: function() {
            // Remove progress bar and redraw
            progressLabel.text( "Finished loading!" );
            g_progressbar.hide();
            redrawSequence();
            g_progressbar.trigger('markercomplete');
            if(g_isPlaying)
                incrementFrame();
      }
    });

    for(var i = 0; i < frames_to_annotate.length; ++i) {
        addKeyFrame(frames_to_annotate[i]);
    }

    // TODO: Ask if user wants to remove just this keyframe or entire subsequence labelling??
    $("#removeFrameButton").click(function() {
        setPlayButton(false);
        if(g_targetFrames.includes(g_currentFrameNr)) {
            g_targetFrames.splice(g_targetFrames.indexOf(g_currentFrameNr), 1);
            g_currentTargetFrameIndex = -1;
            $('#sliderMarker' + g_currentFrameNr).css('background-color', '#888888'); //change color to gray
            //$('#sliderMarker' + g_currentFrameNr).remove();
            $('#selectedFrames' + g_currentFrameNr).remove();
            $('#selectedFramesForm' + g_currentFrameNr).remove();
            g_labels = dictDelete(g_labels, g_currentFrameNr);
            updateFrameLabelVariables();
        }
    });

    $("#nextFrameButton").click(function() {
        goToNextKeyFrame();
    });

    // Moving between frames
    // Scrolling (mouse must be over canvas)
    $("#canvas").bind('mousewheel DOMMouseScroll', function(event){
        g_shiftKeyPressed = event.shiftKey;
        console.log('Mousewheel event!');
        if(event.originalEvent.wheelDelta > 0 || event.originalEvent.detail < 0) {
            // scroll up
            if(g_shiftKeyPressed) {
                goToNextKeyFrame();
            } else {
                goToFrame(g_currentFrameNr + 1);
            }
        } else {
            // scroll down
            if(g_shiftKeyPressed) {
                goToPreviousKeyFrame();
            } else {
                goToFrame(g_currentFrameNr - 1);
            }
        }
        event.preventDefault();
    });

    // Arrow key pressed
    $(document).keydown(function(event){
        g_shiftKeyPressed = event.shiftKey;
        if(event.which === 37) { // Left
            if(g_shiftKeyPressed) {
                goToPreviousKeyFrame();
            } else {
                goToFrame(g_currentFrameNr - 1);
            }
        } else if(event.which === 39) { // Right
            if(g_shiftKeyPressed) {
                goToNextKeyFrame();
            } else {
                goToFrame(g_currentFrameNr + 1);
            }
        }
    });

    $(document).keyup(function(event) {
        g_shiftKeyPressed = event.shiftKey;
    });


    // Load images
    g_framesLoaded = 0;
    //console.log('start: ' + start + ' end: ' + end)
    //console.log('target_frame: ' + target_frame)
    for(var i = start; i <= end; i++) {
        var image = new Image();
        image.src = '/show_frame/' + image_sequence_id + '/' + i + '/' + g_taskID + '/';
        image.onload = function() {
            g_canvasWidth = this.width;
            g_canvasHeight = this.height;
            canvas.setAttribute('width', g_canvasWidth);
            canvas.setAttribute('height', g_canvasHeight);

            // Update progressbar
            g_framesLoaded++;
            g_progressbar.progressbar( "value", g_framesLoaded*100/totalToLoad);
        };
        g_sequence.push(image);
    }
}

function addLabelButton(label_id, label_name,  color_red, color_green, color_blue, parent_id) {
    var labelButton = {
        id: label_id,
        name: label_name,
        red: color_red,
        green: color_green,
        blue: color_blue,
        parent_id: parent_id,
    };
    g_labelButtons.push(labelButton);

    $("#labelButton" + label_id).css("background-color", colorToHexString(color_red, color_green, color_blue));

    // TODO finish
    if(parent_id != 0) {
        $('#sublabel_' + parent_id).hide();
    }
}

function addKeyFrame(frame_nr) {
    if(g_targetFrames.includes(frame_nr)) // Already exists
        return;
    g_targetFrames.push(frame_nr);
    g_targetFrames.sort(function(a, b){return a-b});
    $("#framesSelected").append('<li id="selectedFrames' + frame_nr + '">' + frame_nr + '</li>');
    $("#framesForm").append('<input id="selectedFramesForm' + frame_nr + '" type="hidden" name="frames" value="' + frame_nr + '">');
}

function setupSliderMark(frame, color) {
    color = typeof color !== 'undefined' ? color : '#0077b3';

    let slider = document.getElementById('slider')

    let newMarker = document.createElement('span');
    newMarker.setAttribute('id', 'sliderMarker' + frame);
    $(newMarker).css('background-color', color);
    $(newMarker).css('width', ''+(100.0/g_sequenceLength)+'%');
    $(newMarker).css('margin-left', $('.ui-slider-handle').css('margin-left'));
    $(newMarker).css('height', '100%');
    $(newMarker).css('z-index', '99');
    $(newMarker).css('position', 'absolute');
    $(newMarker).css('left', ''+(100.0*(frame-g_startFrame)/g_sequenceLength)+'%');

    slider.appendChild(newMarker);
    // console.log('Made marker');
}

function sliderMarkSubsequence(frame, frame_end, color) {
    color = typeof color !== 'undefined' ? color : '#0077b3';

    let slider = document.getElementById('slider')

    let newMarker = document.createElement('span');
    newMarker.setAttribute('id', 'sliderMarker' + frame);
    $(newMarker).css('background-color', color);
    $(newMarker).css('width', ''+(100.0*(frame_end-frame)/g_sequenceLength)+'%');
    $(newMarker).css('margin-left', $('.ui-slider-handle').css('margin-left'));
    $(newMarker).css('height', '100%');
    $(newMarker).css('z-index', '99');
    $(newMarker).css('position', 'absolute');
    $(newMarker).css('left', ''+(100.0*(frame-g_startFrame)/g_sequenceLength)+'%');

    slider.appendChild(newMarker);
    // console.log('Made marker');
}

function removeAllSliderMarks() {
    // remove the subsequences but leave the slider handle
    $('#slider').children().not('.ui-slider-handle').remove();

    console.log('Removed all markers');
}

function incrementFrame() {
    if(!g_isPlaying) // If this is set to false, stop playing
        return;
    g_currentFrameNr = ((g_currentFrameNr-g_startFrame) + 1) % g_framesLoaded + g_startFrame;
    var marker_index = g_targetFrames.findIndex(index => index === g_currentFrameNr);
    if(marker_index) {
        g_currentTargetFrameIndex = g_currentFrameNr;
    } else {
        g_currentTargetFrameIndex = -1;
    }
    $('#slider').slider('value', g_currentFrameNr); // Update slider
    $('#currentFrame').text(g_currentFrameNr);

    updateFrameLabelVariables();
    redrawSequence();
    window.setTimeout(incrementFrame, 200);
}

function goToFrame(frameNr) {
    setPlayButton(false);
    g_currentFrameNr = Math.min(Math.max(0, frameNr), g_framesLoaded-1);
    $('#slider').slider('value', frameNr); // Update slider
    $('#currentFrame').text(g_currentFrameNr);
    updateFrameLabelVariables();
    var marker_index = g_targetFrames.findIndex(index => index === frameNr);
    if(marker_index !== -1) {
        g_currentTargetFrameIndex = g_currentFrameNr;
    } else {
        g_currentTargetFrameIndex = -1;
    }
    redrawSequence();
}

function getLabelWithId(id) {
    for(var i = 0; i < g_labelButtons.length; i++) {
        if (g_labelButtons[i].id == id) {
            return g_labelButtons[i];
        }
    }
    return null
}
