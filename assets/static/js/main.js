$(function () {
    var answer = $("#answer");
    answer.hide();

    $("#rnd").click(function () {
        $.ajax({
            method: "GET",
            url: "/random",
        }).done(function (data) {
            $("#paragraph").val(data.paragraph);
            $("#question").val(data.question);
            answer.text("....").hide();
        });
    });

    $("#send").click(function () {
        answer.text("....").show();

        $.ajax({
            method: "POST",
            url: "/predict",
            data: JSON.stringify({
                paragraph: $("#paragraph").val(),
                question: $("#question").val()
            }),
            contentType : "application/json",
        }).done(function (data) {
            answer.text(data.answer);
        });

        return false;
    });

    $('#paragraph, #question').bind('input propertychange', function() {
        answer.text("....").hide();
    });
});
