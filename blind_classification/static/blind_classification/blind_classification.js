
function sendDataForSave() {

    return $.ajax({

        type: "POST",
        url: "/blind_classification/save/",
        data: {
            image_id: g_imageID,
            task_id: g_taskID,
            label_id: g_currentLabel,
            quality: 'unknown',
            rejected: g_rejected ? 'true':'false',
            comments: $('#comments').val(),
            target_frames: JSON.stringify(g_targetFrames),
        },
        dataType: "json" // Need this do get result back as JSON
    });
}

function loadBlindClassificationTask() {
    // Remove slider mark for the last frame
    var lastFrame = g_targetFrames[g_targetFrames.length - 1];
    $('#sliderMarker' + lastFrame).remove();
    console.log('removed slider marker for frame', lastFrame);

    // Add click listener for label buttons to trigger save
    for(var i = 0; i < g_labelButtons.length; ++i) {
        var label_id = g_labelButtons[i].id;
        $('#labelButton' + label_id).click(function() {
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
                g_targetFrames = [];
        });
    }

    console.log("g_targetFrames after:", g_targetFrames);
}
