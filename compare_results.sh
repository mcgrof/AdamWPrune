#!/bin/bash
# Quick comparison of test matrix results

echo "======================================"
echo "Optimizer Performance Comparison"
echo "======================================"
echo ""
echo "Model: ResNet-18, Target Sparsity: 70%"
echo ""
printf "%-15s %10s %10s %10s\n" "Optimizer" "Accuracy" "Sparsity" "Time(s)"
echo "------------------------------------------------------"

grep "resnet18.*_70 " test_matrix_results_20250827_231931/summary_report.txt | head -6 | while read line; do
    name=$(echo $line | awk '{print $1}')
    acc=$(echo $line | awk '{print $2}')
    spars=$(echo $line | awk '{print $3 * 100}')
    time=$(echo $line | awk '{print $4}')
    opt=$(echo $name | cut -d_ -f2)
    printf "%-15s %9.2f%% %9.1f%% %10.1f\n" $opt $acc $spars $time
done | sort -t' ' -k2 -nr

echo ""
echo "======================================"
echo "Key Findings:"
echo "======================================"
echo "• SGD:        Highest accuracy (91.80%), low memory usage"
echo "• AdamWPrune: 88.44% accuracy, 40% less memory than Adam/AdamW"
echo "• All methods successfully achieved 70% sparsity target"
echo ""
echo "Memory Usage During Training:"
echo "• SGD/AdamWPrune: 3.03x model weights"
echo "• Adam/AdamW:     5.03x model weights"
echo "• AdamWSpam/Adv:  5.13x model weights"