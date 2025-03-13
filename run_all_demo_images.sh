for i in $(seq 1 5);
do
	nice autoforge --input_image images/nature.jpg --csv_file bambulab.csv --layer_height 0.04 --pruning_max_colors 8 --pruning_max_swaps 20 --output_folder demo/outputs_nature_${i} &
	sleep 5
	nice autoforge --input_image images/chameleon.jpg --csv_file bambulab.csv --layer_height 0.04 --pruning_max_colors 8 --pruning_max_swaps 20 --output_folder demo/outputs_chameleon_${i} & 
	sleep 5
	nice autoforge --input_image images/cat.jpg --csv_file bambulab.csv --layer_height 0.04 --pruning_max_colors 8 --pruning_max_swaps 20 --output_folder demo/outputs_cat_${i} &
        sleep 5
        nice autoforge --input_image images/stones.jpg --csv_file bambulab.csv --layer_height 0.04 --pruning_max_colors 8 --pruning_max_swaps 20 --output_folder demo/outputs_stones_${i} &
        sleep 10
done
