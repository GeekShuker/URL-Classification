# Reports output

## Tables
- tables/model_comparison.csv (and .md if pandas installed)
- tables/manifest.json

## Plots (reports/plots)
- roc_auc.png, pr_auc.png
- f1.png, accuracy.png, recall.png, precision.png, fpr.png, threshold.png
- fp_count.png, fn_count.png, fp_rate.png, fn_rate.png
- confusion_<model>.png
- timing_breakdown.png
- recall_under_policy.png
- top_features_<model>.png (only if metadata includes top_features)

Note: ROC/PR curves require per-sample scores; this script plots AUC summary values from metadata.
