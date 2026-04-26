const dashboardData = {
  projectTitle: "WavDino - Temporal DashBoard",
  projectSubtitle: "",
  overallHighlights: {
    overallAccuracy: 87.3,
    overallF1: 0.86,
    datasetsCovered: 3,
    bestModel: "Audio-Visual - Temporal (Proposed)"
  },
  datasetPerformance: [
    { dataset: "CREMA-D", samples: 7442, accuracy: 87.8, f1Score: 0.87 },
    { dataset: "RAVDESS", samples: 24144, accuracy: 86.7, f1Score: 0.86 },
    { dataset: "AFEW", samples: 1867, accuracy: 73.1, f1Score: 0.7 }
  ],
  modelComparison: [
    { model: "Audio-Only", accuracy: 84.5, f1Score: 0.83, proposed: false },
    { model: "Visual - Static", accuracy: 83.2, f1Score: 0.82, proposed: false },
    { model: "Audio-Visual - Static", accuracy: 86.0, f1Score: 0.85, proposed: false },
    { model: "Audio-Visual - Temporal (Proposed)", accuracy: 87.3, f1Score: 0.86, proposed: true }
  ],
  crossDatasetGeneralization: [
    { split: "CREMA-D to RAVDESS", accuracy: 85.0, f1Score: 0.84 },
    { split: "CREMA-D to AFEW", accuracy: 71.4, f1Score: 0.69 },
    { split: "RAVDESS to CREMA-D", accuracy: 83.8, f1Score: 0.83 }
  ],
  confusionMatrix: {
    labels: ["Happy", "Sad", "Angry", "Neutral"],
    values: [
      [0.92, 0.04, 0.03, 0.01],
      [0.05, 0.89, 0.04, 0.02],
      [0.04, 0.04, 0.9, 0.02],
      [0.03, 0.03, 0.03, 0.91]
    ]
  }
};

export default dashboardData;
