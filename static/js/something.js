var input1 = document.getElementById('upload1');
var infoArea1 = document.getElementById('upload1-label');

var input2 = document.getElementById('upload2');
var infoArea2 = document.getElementById('upload2-label');

var input0 = document.getElementById('upload0');
var infoArea0 = document.getElementById('upload0-label');

var inputJ = document.getElementById('uploadJson');
var infoAreaJ = document.getElementById('uploadJson-label');

input0.addEventListener('change', showFileName0);
inputJ.addEventListener('change', showFileNameJ);
input1.addEventListener('change', showFileName1);
input2.addEventListener('change', showFileName2);


/*  ==========================================
    SHOW UPLOADED IMAGE NAME
* ========================================== */

function showFileName0(event) {
    var input = event.srcElement;
    var fileName = input.files[0].name;
    infoArea0.textContent = 'MRI File: ' + fileName;
}
function showFileNameJ(event) {
    var input = event.srcElement;
    var fileName = input.files[0].name;
    infoAreaJ.textContent = 'Json File: ' + fileName;
}


function showFileName1(event) {
    var input = event.srcElement;
    var fileName = input.files[0].name;
    infoArea1.textContent = 'First MRI: ' + fileName;
}
function showFileName2(event) {
    var input = event.srcElement;
    var fileName = input.files[0].name;
    infoArea2.textContent = 'Second MRI: ' + fileName;
}


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



/*  ==========================================
    same anorher fx
* ========================================== */
function showHide(secim) {
    $("#" + secim).toggle();
}






/*  ==========================================
    same anorher
* ========================================== */



$(function () {
   /* showHide('preOrj1');
    showHide('preOrj2');
    showHide('preDetec1'); 
    showHide('preDetec2');*/
    $('#upload0').on('change', function () {
        readURL(input0);
    });
    $('#upload1').on('change', function () {
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

