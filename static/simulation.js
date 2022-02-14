$('#appForm').submit(function(event){        
    event.preventDefault(); // avoid to execute the actual submit of the form.

    var form = $(this);
    var actionUrl = form.attr('action');
    
    
    // Run Python script
    jqXHR = $.ajax({
        type: "POST",
        url: actionUrl,
        data: form.serialize(), // serializes the form's elements.
        async: false,
        success: function (data) {
            console.log('Submission was successful.');
            console.log(data);
        },
        error: function (data) {
            console.log('An error occurred.');
            console.log(data);
        }        
    });
    
    score = parseInt(jqXHR.responseText,10);
    threshold = parseInt($('#approval').val(),10);
    min = parseInt($('#minimum').val(),10);
    max = parseInt($('#maximum').val(),10);
    
    $('#score').hide();
    $('#message').hide();
    $('.loader').show();
    
    setTimeout(() => {                     
        $('.loader').hide();     
        $('#intro').text("Your estimated credit score is...")   
        if (score >= threshold) {
            $('#score').text(score);            
            $('#score').css("color","green");            
            chance = (score-threshold)/(max-threshold)*100;
            $("#message").text("Congratulations, your approval chance is: "+chance.toFixed(1)+"%!");
        } else {
            $('#score').text(score);            
            $('#score').css("color","red");            
            chance = (threshold-score)/(threshold-min)*100;
            $("#message").text("Unfortunately, your rejection chance is: "+chance.toFixed(1)+"%.");
        }
        $('#score').show();
        $('#message').show();
    }, 1000);     
});