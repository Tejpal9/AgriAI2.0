const express = require("express");
const router = express.Router();

router.get("/alert", (req, res) => {
  res.render("weather/weatherAlert.ejs", { tempAlert: null });
});

router.post("/alert", async (req, res) => {
  try {
    const city = req.body.city;

    if (!city) {
      return res.status(400).send({ error: "City name is required." });
    }

    // 1. Geocoding
    const geoUrl = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=en&format=json`;
    const geoResponse = await fetch(geoUrl);
    const geoData = await geoResponse.json();

    if (!geoData.results || geoData.results.length === 0) {
      return res.status(404).send({ error: "City not found." });
    }

    const { latitude, longitude } = geoData.results[0];

    // 2. Weather Data
    const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,rain,snowfall&forecast_days=1`;
    const weatherResponse = await fetch(weatherUrl);
    const weatherData = await weatherResponse.json();

    const current = weatherData.current;
    
    let resObject = {
      city: city,
      temp: current.temperature_2m,
      humidity: current.relative_humidity_2m,
      rain: current.rain,
      snow: current.snowfall
    };

    let tempAlert, tempmsg, humAlert, humMsg, rainAlert, rainMsg;

    // --- Temperature Logic ---
    if (resObject.temp > 35) {
      tempAlert = `High Temperature (${resObject.temp}°C)`;
      tempmsg = [
        "Avoid excessive watering.",
        "Consider shade nets or mulching.",
        "Monitor for heat stress.",
      ];
    } else if (resObject.temp < 15) {
      tempAlert = `Low Temperature (${resObject.temp}°C)`;
      tempmsg = [
        "Protect crops from frost.",
        "Consider greenhouse cultivation.",
        "Monitor for chilling injury.",
      ];
    } else {
      // ADDED: Logic for Normal Temperature
      tempAlert = `Temperature is Good (${resObject.temp}°C)`;
      tempmsg = [
        "Conditions are optimal for growth.",
        "Continue standard irrigation schedule.",
        "No special thermal protection needed."
      ];
    }

    // --- Humidity Logic ---
    if (resObject.humidity > 80) {
      humAlert = `High Humidity (${resObject.humidity}%)`;
      humMsg = [
        "Improve drainage to prevent waterlogging.",
        "Space plants for air circulation.",
        "Monitor for fungal diseases.",
      ];
    } else if (resObject.humidity < 40) {
      humAlert = `Low Humidity (${resObject.humidity}%)`;
      humMsg = [
        "Increase watering frequency.",
        "Consider misting.",
        "Monitor for wilting.",
      ];
    } else {
      // ADDED: Logic for Normal Humidity
      humAlert = `Humidity is Good (${resObject.humidity}%)`;
      humMsg = [
        "Moisture levels are balanced.",
        "Maintain regular monitoring.",
        "Good environment for photosynthesis."
      ];
    }

    // --- Rain Logic ---
    if (resObject.rain > 0) {
      rainAlert = "Rainy Weather";
      rainMsg = [
        "Check for waterlogging.",
        "Ensure proper drainage.",
        "Watch for fungal issues.",
      ];
    } else if (resObject.snow > 0) {
      rainAlert = "Snow / Freezing";
      rainMsg = [
        "Protect crops from heavy snow.",
        "Provide insulation.",
      ];
    } else {
      // ADDED: Logic for No Rain
      rainAlert = "No Rain Detected";
      rainMsg = [
        "Weather is dry.",
        "Ensure soil moisture is maintained manually."
      ];
    }

    res.send({
      tempAlert, tempmsg,
      humAlert, humMsg,
      rainAlert, rainMsg,
      city,
    });

  } catch (e) {
    console.error(e);
    res.status(500).send({ error: "Server Error" });
  }
});

module.exports = router;