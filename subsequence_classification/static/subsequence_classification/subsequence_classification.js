var g_backgroundImage;
var g_frameNr;
var g_currentColor = null;
var g_labels = {}; // Dictionary with keys frame_nr which each has a label
var g_currentFrame = -1;


// TODO: Modify function. For now, copied from classification.js
function loadSubsequenceClassificationTask() {
    console.log('In load subsequence classification')

    $('#clearButton').click(function() {
        g_annotationHasChanged = true;
        // Reset image quality form
        $('#imageQualityForm input[type="radio"]').each(function(){
            $(this).prop('checked', false);
        });
        // Set all label buttons to inactive
        changeLabel(-1);
    });

    // Add click listener for label buttons to trigger save
    for(var i = 0; i < g_labelButtons.length; ++i) {
        var label_id = g_labelButtons[i].id;
        $('#labelButton' + label_id).click(function() {
            // TODO: Implement functionality when clicking a label button here


            /*  From classification task
            // Only trigger save if label has no children
            var childrenFound = false;
            for(var j = 0; j < g_labelButtons.length; j++) {
                var child_label = g_labelButtons[j];
                if(child_label.parent_id == g_currentLabel) {
                    childrenFound = true;
                } else if(child_label.parent_id != 0) {
                }
            }
            if(!childrenFound)
                save();
            */

        });
    }

    setupSubsequenceClassification();
}


function setupSubsequenceClassification() {
    console.log('Setting up subsequence classification....');

    redrawSequence();
}


// TODO: Modify function. For now, copied from classification.js
function sendDataForSave() {
    return $.ajax({
        type: "POST",
        url: "/subsequence_classification/save/",
        data: {
            image_id: g_imageID,
            task_id: g_taskID,
            label_id: g_currentLabel,
            quality: $('input[name=quality]:checked').val(),
            rejected: g_rejected ? 'true':'false',
            comments: $('#comments').val(),
            target_frames: JSON.stringify(g_targetFrames),
        },
        dataType: "json" // Need this do get result back as JSON
    });
}

