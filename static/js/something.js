var input1 = document.getElementById('upload');
var infoArea = document.getElementById('upload-label');

var input2 = document.getElementById('upload2');
var infoArea2 = document.getElementById('upload2-label');

input1.addEventListener('change', showFileName);
input2.addEventListener('change', showFileName2);


/*  ==========================================
    SHOW UPLOADED IMAGE
* ========================================== */



function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        input.messages = ""
        reader.onload = function (e) {
            $('#imageResult')
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
        
    }
}
function readURL2(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#imageResult2')
                .attr('src', e.target.result);
        };
        input.messages = ""
        reader.readAsDataURL(input.files[0]);
    }

}

$(function () {
    $('#upload').on('change', function () {
        readURL(input1);
    });

    $('#upload2').on('change', function () {
        readURL2(input2);
    });
    $("form[name='baseForm']").validate({
        rules: {
            // The key name on the left side is the name attribute
            // of an input field. Validation rules are defined
            // on the right side
            firstMR: "required",
            secondMR: "required"
        },
        messages: {
            firstMR: "please select a file ",
            secondMR: "please select a file"
        },
        // Make sure the form is submitted to the destination defined
        // in the "action" attribute of the form when valid
        submitHandler: function (form) {
            form.submit();
        }
    });
});


/*  ==========================================
    SHOW UPLOADED IMAGE NAME
* ========================================== */



function showFileName(event) {
    var input = event.srcElement;
    var fileName = input.files[0].name;
    infoArea.textContent = 'First MRI: ' + fileName;
}
function showFileName2(event) {
    var input = event.srcElement;
    var fileName = input.files[0].name;
    infoArea2.textContent = 'Second MRI: ' + fileName;
}