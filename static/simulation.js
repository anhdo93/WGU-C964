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

    result = jqXHR.responseText;
    score = Math.round((1-result)*300+400);

    $('#score').hide();
    $('.loader').show();

    setTimeout(() => {                     
        $('.loader').hide();        
        $('#score').text(score);            
        $('#score').show();
    }, 1000);     
});