var input_file;
var reader = new FileReader();

$("#btn_upload").click(function (e) {
  e.preventDefault();
  $($("#input_file")[0]).trigger("click");
});

$("#btn_clear").click(function (e) {
  clear_canvas();
});

$("#input_file").change(function (ev) {
  ev.preventDefault();
  input_file = $("#input_file").prop("files")[0];
  reader.onload = function (event) {
    document.getElementById("image_raw").className = "visible";
    $("#image_raw").attr("src", event.target.result);
  };
  reader.readAsDataURL(input_file);
  clear_canvas();
  // $(".result").hide();
});

//#? bad
document
  .getElementById("input_file")
  .addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        const imageRaw = document.getElementById("image_raw");
        imageRaw.src = e.target.result;
        imageRaw.style.display = "block";
        document.querySelector(".placeholder-img").style.display = "none"; // Hide the placeholder
      };
      reader.readAsDataURL(file);
    }
  });

$("#btn_process").on("click", function (ev) {
  ev.preventDefault();

  canvas = $("#drawing")[0];
  mask_b64 = canvas.toDataURL("image/png");
  form_data = new FormData();
  form_result = $("form")[0];

  form_data.append("input_file", input_file);
  form_data.append("mask_b64", mask_b64);
  form_data.forEach(function (value, key) {
    console.log(key, value);
  });

  $.ajax({
    url: "/process-image",
    type: "post",
    contentType: "application/json",
    data: form_data,
    processData: false,
    contentType: false,
    crossDomain: true,
    cache: false,
    beforeSend: function () {
      $(".overlay").show(); // loading spinner
      document.querySelector(".overlay").style.display = "flex";
      $("#div-result").html("");
      // $(".result").hide();
    },
  })
    .done(function (jsondata, textStatus, jqXHR) {
      image = jsondata["output_image"];
      image = window.location + "/" + image;
      $("#div-result").append(
        `<img id='image_output' src="${image}" class="img_result" width=384 height=384>`
      );

      $(".overlay").hide();
      $(".result").css("display", "flex");
    })
    .fail(function (jsondata, textStatus, jqXHR) {
      alert(jsondata["responseJSON"]);
      $(".overlay").hide();
    });
});

const dropZone = document.getElementById("drop-zone");
const imageRaw = document.getElementById("image_raw");

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");

  const file = event.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imageRaw.src = e.target.result;
      document.getElementById("image_raw").className = "visible";
    };
    reader.readAsDataURL(file);

    input_file = file;

    clear_canvas();
    // $(".result").hide();
  }
});

const wrapper = document.querySelector(".wrapper");

wrapper.addEventListener("dragover", (event) => {
  event.preventDefault();
  wrapper.classList.add("dragging");
});

wrapper.addEventListener("dragleave", () => {
  wrapper.classList.remove("dragging");
});

wrapper.addEventListener("drop", (event) => {
  event.preventDefault();
  wrapper.classList.remove("dragging");

  const file = event.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imageRaw.src = e.target.result;
      document.getElementById("image_raw").className = "visible";
    };
    reader.readAsDataURL(file);

    input_file = file;

    clear_canvas();
    // $(".result").hide();
  }
});

document.getElementById("image_raw").addEventListener("load", function () {
  document.querySelector(".placeholder-img").style.display = "none"; // Hide placeholder
});

const themeSwitch = document.getElementById("theme-switch");

if (localStorage.getItem("theme") === "dark") {
  document.body.classList.add("dark-mode");
  themeSwitch.checked = true;
}

themeSwitch.addEventListener("change", () => {
  if (themeSwitch.checked) {
    document.body.classList.add("dark-mode");
    localStorage.setItem("theme", "dark");
  } else {
    document.body.classList.remove("dark-mode");
    localStorage.setItem("theme", "light");
  }
});