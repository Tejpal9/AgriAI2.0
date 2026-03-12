const express = require("express");
const router = express.Router();

// --- Database Configuration ---
const plantDatabase = {
  tomato: {
    seedling: { N: 20, P: 10, K: 15 },
    flowering: { N: 15, P: 15, K: 20 },
    fruiting: { N: 10, P: 20, K: 30 },
  },
  wheat: {
    seedling: { N: 25, P: 5, K: 10 },
    mature: { N: 15, P: 10, K: 15 },
  },
  corn: {
    seedling: { N: 30, P: 15, K: 25 },
    flowering: { N: 20, P: 20, K: 30 },
    mature: { N: 10, P: 15, K: 35 },
  },
  potato: {
    seedling: { N: 18, P: 10, K: 18 },
    flowering: { N: 15, P: 12, K: 20 },
    harvest: { N: 12, P: 18, K: 25 },
  },
  rice: {
    seedling: { N: 22, P: 8, K: 12 },
    flowering: { N: 18, P: 14, K: 20 },
    mature: { N: 15, P: 10, K: 18 },
  },
  cabbage: {
    seedling: { N: 15, P: 10, K: 12 },
    flowering: { N: 10, P: 12, K: 18 },
    mature: { N: 8, P: 15, K: 20 },
  },
  carrot: {
    seedling: { N: 20, P: 10, K: 15 },
    flowering: { N: 18, P: 12, K: 18 },
    harvest: { N: 15, P: 18, K: 25 },
  },
  lettuce: {
    seedling: { N: 10, P: 8, K: 10 },
    flowering: { N: 12, P: 10, K: 12 },
    mature: { N: 8, P: 10, K: 10 },
  },
  spinach: {
    seedling: { N: 15, P: 10, K: 10 },
    flowering: { N: 12, P: 10, K: 12 },
    mature: { N: 10, P: 12, K: 15 },
  },
  onion: {
    seedling: { N: 18, P: 10, K: 15 },
    flowering: { N: 15, P: 12, K: 18 },
    mature: { N: 12, P: 15, K: 20 },
  },
  garlic: {
    seedling: { N: 12, P: 10, K: 10 },
    flowering: { N: 10, P: 12, K: 15 },
    harvest: { N: 8, P: 15, K: 18 },
  },
  peas: {
    seedling: { N: 20, P: 12, K: 15 },
    flowering: { N: 18, P: 14, K: 18 },
    mature: { N: 15, P: 18, K: 20 },
  },
  beans: {
    seedling: { N: 22, P: 10, K: 15 },
    flowering: { N: 18, P: 12, K: 20 },
    mature: { N: 15, P: 18, K: 25 },
  },
  strawberry: {
    seedling: { N: 15, P: 10, K: 12 },
    flowering: { N: 12, P: 12, K: 15 },
    fruiting: { N: 10, P: 15, K: 18 },
  },
  blueberry: {
    seedling: { N: 12, P: 10, K: 8 },
    flowering: { N: 10, P: 12, K: 10 },
    fruiting: { N: 8, P: 12, K: 12 },
  },
  apple: {
    seedling: { N: 30, P: 20, K: 25 },
    flowering: { N: 25, P: 15, K: 20 },
    fruiting: { N: 20, P: 25, K: 30 },
  },
  peach: {
    seedling: { N: 25, P: 20, K: 25 },
    flowering: { N: 20, P: 25, K: 30 },
    fruiting: { N: 15, P: 30, K: 35 },
  },
  apricot: {
    seedling: { N: 20, P: 15, K: 20 },
    flowering: { N: 18, P: 18, K: 22 },
    fruiting: { N: 15, P: 20, K: 25 },
  },
  pear: {
    seedling: { N: 25, P: 15, K: 20 },
    flowering: { N: 20, P: 20, K: 25 },
    fruiting: { N: 15, P: 25, K: 30 },
  },
  cherry: {
    seedling: { N: 30, P: 25, K: 20 },
    flowering: { N: 25, P: 20, K: 25 },
    fruiting: { N: 20, P: 30, K: 35 },
  },
  grape: {
    seedling: { N: 20, P: 15, K: 15 },
    flowering: { N: 18, P: 18, K: 20 },
    fruiting: { N: 15, P: 20, K: 25 },
  },
  mango: {
    seedling: { N: 25, P: 15, K: 25 },
    flowering: { N: 20, P: 20, K: 30 },
    fruiting: { N: 15, P: 25, K: 35 },
  },
  banana: {
    seedling: { N: 30, P: 20, K: 30 },
    flowering: { N: 25, P: 25, K: 35 },
    fruiting: { N: 20, P: 30, K: 40 },
  },
};

const diseaseAdjustments = {
  powdery_mildew: { N: 1.2 },
  leaf_spot: { K: 1.2 },
  root_rot: { N: 1.3 },
  aphids: { P: 1.15 },
  blight: { N: 1.25, P: 1.1 },
  rust: { K: 1.2 },
  downy_mildew: { N: 1.15 },
  botrytis: { K: 1.1 },
  fruit_rot: { P: 1.2 },
  whitefly: { N: 1.1 },
  canker: { P: 1.15 },
  tomato_spot: { K: 1.25 },
  fungus: { N: 1.1 },
  leaf_blight: { N: 1.2, K: 1.2 },
  thrips: { P: 1.2 },
  powdery_fungus: { N: 1.25 },
  downy_fungus: { P: 1.25 },
  leaf_mold: { K: 1.3 },
  rot: { N: 1.4 },
  yellowing: { P: 1.3 },
  stem_borer: { K: 1.15 },
  plant_virus: { N: 1.2 },
  black_spot: { P: 1.1 },
  rust_disease: { K: 1.3 },
  wilting: { N: 1.25 },
  gray_mold: { K: 1.2 },
  stem_rot: { P: 1.2 },
  leaf_curl: { N: 1.15 },
  mosaic_virus: { P: 1.25 },
  cherry_leaf_spot: { K: 1.1 },
  wilt_disease: { N: 1.3 },
  citrus_greening: { P: 1.3 },
  leaf_scorch: { K: 1.25 },
  gummosis: { N: 1.1 },
  pseudomonas: { P: 1.15 },
  brown_spot: { K: 1.2 },
  root_knot: { N: 1.2 },
  mildew: { K: 1.15 },
  yellow_mold: { P: 1.2 },
  fungal_spot: { N: 1.3 },
  pest_infested: { P: 1.2 },
  leaf_disease: { K: 1.25 },
  botrytis_blight: { N: 1.3 },
  spider_mite: { P: 1.15 },
  virus_infection: { K: 1.15 },
  root_disease: { N: 1.25 },
  
  // Specific/Complex Keys (retained for compatibility)
  Apple__Apple_scab: { N: 1.2, P: 1.1, K: 1.15 },
  Apple_Black_rot: { N: 1.25, P: 1.2 },
  Apple_Cedar_apple_rust: { N: 1.1, P: 1.2, K: 1.1 },
  Apple__healthy: {},
  Blueberry__healthy: {},
  "Cherry(including_sour)_Powdery_mildew": { N: 1.2, P: 1.1 },
  "Cherry_(including_sour)healthy": {},
  "Corn(maize)_Cercospora_leaf_spot_Gray_leaf_spot": { N: 1.3, P: 1.2 },
  "Corn_(maize)Common_rust": { N: 1.25, P: 1.15, K: 1.1 },
  "Corn_(maize)Northern_Leaf_Blight": { N: 1.2, P: 1.3 },
  "Corn(maize)_healthy": {},
  Grape__Black_rot: { N: 1.2, P: 1.2 },
  "Grape_Esca(Black_Measles)": { N: 1.3, P: 1.25, K: 1.15 },
  "Grape__Leaf_blight(Isariopsis_Leaf_Spot)": { N: 1.2, P: 1.3, K: 1.2 },
  Grape__healthy: {},
  "Orange_Haunglongbing(Citrus_greening)": { N: 1.3, P: 1.2 },
  Peach___Bacterial_spot: { N: 1.2, P: 1.1 },
  Peach__healthy: {},
  Pepper_bell_Bacterial_spot: { N: 1.2, P: 1.1, K: 1.2 },
  Pepper_bell__healthy: {},
  Potato__Early_blight: { N: 1.2, P: 1.25 },
  Potato_Late_blight: { N: 1.3, P: 1.3, K: 1.1 },
  Potato__healthy: {},
  Raspberry__healthy: {},
  Soybean_healthy: {},
  Squash__Powdery_mildew: { N: 1.1, P: 1.2 },
  Strawberry__Leaf_scorch: { N: 1.15, P: 1.25, K: 1.2 },
  Strawberry_healthy: {},
  Tomato__Bacterial_spot: { N: 1.2, P: 1.2 },
  Tomato__Early_blight: { N: 1.2, P: 1.25 },
  Tomato_Late_blight: { N: 1.3, P: 1.3, K: 1.1 },
  Tomato__Leaf_Mold: { N: 1.15, P: 1.2 },
  Tomato__Septoria_leaf_spot: { N: 1.1, P: 1.25 },
  Tomato__Spider_mites_Two_spotted_spider_mite: { N: 1.2, P: 1.15, K: 1.1 },
  Tomato__Target_Spot: { N: 1.25, P: 1.2 },
  Tomato_Tomato_Yellow_Leaf_Curl_Virus: { N: 1.2, P: 1.2, K: 1.25 },
  Tomato__Tomato_mosaic_virus: { N: 1.3, P: 1.1 },
  Tomato___healthy: {},
};

// --- Logic Function ---
function getFertilizerRecommendation(plantInput, growthStageInput, diseaseInput = null, soilData = {}) {
  // 1. Normalize Inputs (Lowercase & Trim)
  // This allows "Tomato" to match "tomato" in the database
  const plant = plantInput ? plantInput.toLowerCase().trim() : "";
  const growthStage = growthStageInput ? growthStageInput.toLowerCase().trim() : "";
  
  // Check if plant and growth stage exist in DB
  if (!plantDatabase[plant] || !plantDatabase[plant][growthStage]) {
    return null; // Return null if not found
  }

  // Clone the base nutrients to avoid modifying the database
  const nutrients = { ...plantDatabase[plant][growthStage] };

  // 2. Handle Disease Adjustment
  if (diseaseInput) {
    // Try to find an exact match first
    let diseaseKey = Object.keys(diseaseAdjustments).find(
      key => key.toLowerCase() === diseaseInput.toLowerCase().trim()
    );

    // If no exact match, try matching by replacing spaces with underscores (e.g., "Root Rot" -> "root_rot")
    if (!diseaseKey) {
       const snakeCaseInput = diseaseInput.toLowerCase().trim().replace(/\s+/g, '_');
       if (diseaseAdjustments[snakeCaseInput]) {
         diseaseKey = snakeCaseInput;
       }
    }

    // Apply adjustments if a valid key was found
    if (diseaseKey && diseaseAdjustments[diseaseKey]) {
      for (const [nutrient, factor] of Object.entries(diseaseAdjustments[diseaseKey])) {
        nutrients[nutrient] = (nutrients[nutrient] || 0) * factor;
      }
    }
  }

  // 3. Handle Soil Data Adjustment
  // If soil is already low in a nutrient (< 10), we increase the recommended dose
  if (soilData) {
    if ((soilData.N || 0) < 10) nutrients.N *= 1.1;
    if ((soilData.P || 0) < 10) nutrients.P *= 1.1;
    if ((soilData.K || 0) < 10) nutrients.K *= 1.1;
  }

  // 4. Round values for display
  for (const key in nutrients) {
    nutrients[key] = Math.round(nutrients[key] * 100) / 100;
  }

  return nutrients;
}

// --- Routes ---

// Render the page
router.get("/", (req, res) => {
  res.render("firtilizerCalculator/firtilizer.ejs");
});

// Handle the calculation
router.post("/", (req, res) => {
  const { plant, growthStage, disease, soilData } = req.body;

  const recommendation = getFertilizerRecommendation(
    plant,
    growthStage,
    disease,
    soilData
  );

  // If recommendation is null, it means Plant or Stage wasn't found
  if (!recommendation) {
    return res.status(404).json({ 
      error: "Could not find data for this plant or growth stage. Please try 'Tomato' and 'Flowering'." 
    });
  }

  // Send success response
  res.json(recommendation);
});

module.exports = router;