let g_backgroundImage;
let g_frameNr;
let g_currentColor = null;
let g_labels = {}; // Dictionary with keys frame_nr which each has a label
let g_currentFrameLabelId = -1;
let g_currentFrameLabel = null;
let g_currentFrameLabelColor = '#3d6ad8';


// TODO: Modify function. For now, copied from classification.js
function loadSubsequenceClassificationTask() {
    console.log('In load subsequence classification')

    // Add click listener for label buttons to trigger save
    for (let i = 0; i < g_labelButtons.length; ++i) {
        let label_id = g_labelButtons[i].id;
        $('#labelButton' + label_id).click(function () {
            // TODO: Implement functionality when clicking a label button here
            console.log('labelButton with id ' + $(this).id + ' was clicked');
            console.log($(this));

            // Trigger pop-up with "Start/End segment"
            let dialogDiv = document.createElement('div');
            dialogDiv.className = 'smallFrame';

            // If this segment was started with the same label
            //  --> only allow "End"
            // If this segment was started with another label
            //  --> only allow start
            //  --> start a new segment with the other label
            //  --> also end the previous segment on the previous frame

        });
    }

    setupSubsequenceClassification();

    const startSubsequence = document.querySelector('.startSubsequenceButton');
    const endSubsequence = document.querySelector('.endSubsequenceButton');
    startSubsequence.onclick = startButtonClick;
    endSubsequence.onclick = endButtonClick;
}

function sendDataForSave() {
    return $.ajax({
        type: "POST",
        url: "/subsequence-classification/save/",
        data: {
            image_id: g_imageID,
            task_id: g_taskID,
            frame_labels: JSON.stringify(g_labels),
            target_frames: JSON.stringify(g_targetFrames),
            quality: $('input[name=quality]:checked').val() || 'unknown', // Default to unknown if no quality is selected
            rejected: g_rejected ? 'true':'false',
            comments: $('#comments').val(),
        },
        dataType: "json" // Need this do get result back as JSON
    });
}

function startButtonClick(e) {

    if (g_currentLabel === -1) {
        if (g_labelButtons.length === 1) {
            changeLabel(g_labelButtons[0].id)
            console.log('only one label button')
        }
        else {
            alert('You need to select a label before marking a subsequence!');
            return;
        }
    }

    console.log('Subsequence for', getLabelWithId(g_currentLabel).name, 'started on frame nr', g_currentFrameNr);

    // Label current frame as first in sequence
    $('#currentFrame').text(g_currentFrameNr);
    addKeyFrame(g_currentFrameNr);
    setLabel(g_currentFrameNr, g_currentLabel);

    // Find next frame belonging to a different sequence (if any)
    let nextFrameWithDifferentLabel = findNextFrameWithDifferentLabel(g_currentFrameNr);
    if (nextFrameWithDifferentLabel === -1) {
        console.log('No frame found with different label');
    }
    let lastFrame = g_startFrame + g_sequenceLength;
    let frameIdx = g_currentFrameNr;
    while (frameIdx  < min(nextFrameWithDifferentLabel, lastFrame)) {
        frameIdx++;
        //addKeyFrame(frameIdx);
        setLabel(frameIdx, g_currentLabel);
    }

    updateFrameLabelVariables();
}

function endButtonClick(e) {
    if (!g_currentLabel) {
        alert('You need to select a label before marking a subsequence!');
        return;
    }

    if (jQuery.isEmptyObject(g_labels)) {
        alert('You need to start a subsequence before ending it!');
        return;
    }

    console.log('Subsequence for', g_currentLabel, 'ended on frame nr', g_currentFrameNr);

    // Find start of subsequence with same label
    let startOfSubsequence = findPreviousFrameWithSameLabel(g_currentFrameNr, g_currentLabel);
    let lastFrame = g_startFrame + g_sequenceLength;
    let frameIdx = startOfSubsequence;
    while (frameIdx <= min(g_currentFrameNr, lastFrame)) {
        // console.log('Add label', g_currentLabel, 'to frame', frameIdx);
        addKeyFrame(frameIdx);
        setLabel(frameIdx, g_currentLabel);
        updateFrameLabelVariables();
        frameIdx++;
    }
    // Add one mark for entire subsequence
    sliderMarkSubsequence(startOfSubsequence, min(g_currentFrameNr, lastFrame), g_currentLabel);

    // Label the current frame as the last in the subsequence
    // frameIdx = g_currentFrameNr;
    // addKeyFrame(frameIdx);
    // setLabel(frameIdx, g_currentLabel);
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

function setLabel(frame_nr, label_id) {
    // Set label for frame
    g_labels[frame_nr] = label_id;

    // Update slider marker for frame
    let label = getLabelWithId(label_id);
    let hexColor =  colorToHexString(label.red, label.green, label.blue);
    setupSliderMark(frame_nr, hexColor);
}

function addSubsequenceLabel(frame_nr, label_id) {
    addKeyFrame(frame_nr);
    setLabel(frame_nr, label_id);
}


function updateFrameLabelVariables() {
    // Update g_currentFrameLabelId/g_currentFrameLabel if current frame is labelled
    if (g_currentFrameNr in g_labels) {
        g_currentFrameLabelId = g_labels[g_currentFrameNr];
    } else {
        g_currentFrameLabelId = -1;
    }
    g_currentFrameLabel = getLabelWithId(g_currentFrameLabelId);

    // If current frame is labelled, display label name
    if (g_currentFrameLabel) {
        $('#currentFrameLabel').text(g_currentFrameLabel.name);
        //$('#currentFrameLabel').innerHTML = '<span style="color: ' + g_currentFrameLabelColor + '">' + getLabelWithId(g_currentFrameLabelId).name + '</span>';;
        $('#currentFrameLabelDisplay').text(g_currentFrameLabel.name);
    } else {
        $('#currentFrameLabel').text('No label');
        $('#currentFrameLabelDisplay').text('No label');
    }
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
            $('#sliderMarker' + g_currentFrameNr).remove();
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
    window.setTimeout(incrementFrame, 25);
}

function goToFrame(frameNr) {
    setPlayButton(false);
    g_currentFrameNr = min(max(0, frameNr), g_framesLoaded-1);
    $('#slider').slider('value', frameNr); // Update slider
    $('#currentFrame').text(g_currentFrameNr);
    updateFrameLabelVariables();
    var marker_index = g_targetFrames.findIndex(index => index === frameNr);
    if(marker_index) {
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
