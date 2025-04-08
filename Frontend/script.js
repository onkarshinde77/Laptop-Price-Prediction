// this methos is used to fill the select optionls
function fillSelect(id, options) {
    const select = document.getElementById(id);
    select.innerHTML = "";
    options.forEach((opt) => {
        const option = document.createElement("option");
        option.value = option.text = opt;
        select.appendChild(option);
    });
}

// this method is used to fill the select options with the data from the server
// it is called when the page loads
async function fetchOptions() {
    const res = await fetch("http://localhost:5000/options");
    const data = await res.json();
    fillSelect("brand", data.Company);
    fillSelect("type", data.TypeName);
    fillSelect("ram", data.Ram);
    fillSelect("cpu", data["Cpu brand"]);
    fillSelect("hdd", data.HDD);
    fillSelect("ssd", data.SSD);
    fillSelect("gpu", data.Gpu);
    fillSelect("os", data.OpSys);
}

// this method is used to show the toast(warning) message
function showToast(message) {
    const toast = document.getElementById("toast");
    toast.innerText = message;
    toast.style.display = "block";
    setTimeout(() => {
        toast.style.display = "none";
    }, 3000);
}


function setError(id, message) {
    const el = document.getElementById(id);
    el.classList.add("input-error");
    showToast(message);
}

function clearErrors() {
    document.querySelectorAll(".input-error").forEach((el) => {
        el.classList.remove("input-error");
    });
}

//calculate the PPI of the screen using resolution and screen size from input //
function getPPI() {
    const screen = parseFloat(document.getElementById("screen").value);
    const resolution = document.getElementById("resolution").value.trim();

    if (!resolution.includes("x")) return null;
    const [xStr, yStr] = resolution.toLowerCase().split("x");
    const x = parseInt(xStr);
    const y = parseInt(yStr);
    if (isNaN(screen) || isNaN(x) || isNaN(y)) return null;

    return Math.sqrt(x * x + y * y) / screen;
}

async function predictPrice() {
    clearErrors();
    const loader = document.getElementById("loader");
    loader.style.display = "block";

    const payload = {
        Company: document.getElementById("brand").value,
        TypeName: document.getElementById("type").value,
        Ram: parseInt(document.getElementById("ram").value),
        Weight: parseFloat(document.getElementById("weight").value),
        Touchscreen: document.getElementById("touchscreen").value === "Yes" ? 1 : 0,
        "IPS-Panel": document.getElementById("ips").value === "Yes" ? 1 : 0,
        PPI: getPPI(),
        "Cpu brand": document.getElementById("cpu").value,
        HDD: parseInt(document.getElementById("hdd").value),
        SSD: parseInt(document.getElementById("ssd").value),
        Gpu: document.getElementById("gpu").value,
        OpSys: document.getElementById("os").value,
    };

    // Validate inputs
    if (isNaN(payload.Ram) || payload.Ram <= 0) {
        setError("ram", "Please enter a valid RAM value.");
        loader.style.display = "none";
        return;
    } if (isNaN(payload.Weight) || payload.Weight <= 0) {
        setError("weight", "Please enter a valid weight.");
        loader.style.display = "none";
        return;
    } if (payload.PPI === null || payload.PPI <= 0) {
        setError("resolution", "Invalid screen resolution or screen size.");
        loader.style.display = "none";
        return;
    } if (payload.OpSys === "macOS" && payload.Company !== "Apple") {
        setError("os", "Only Apple laptops support macOS.");
        loader.style.display = "none";
        return;
    } if (payload.Company === "Apple") {
        fillSelect("os", ["macOS"]);
        payload.OpSys = "macOS";
    }

    // send all data to the server for prediction //
    const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    const result = await res.json();
    console.log(result);
    loader.style.display = "none";

    document.getElementById("output").innerText = `Predicted Price: ₹${result.price}`;
    showToast("Prediction Complete!");

}

fetchOptions();
