reset                                   # reset
set size ratio 0.2                      # set relative size of plots
set xlabel 'Sample #'                   # set x-axis label for all plots
set grid xtics ytics                    # grid: enable both x and y lines
set grid lt 1 lc rgb '#cccccc' lw 1     # grid: thin gray lines
set terminal pngcairo enhanced font "Arial,12" size 1200,800 enhanced linewidth 2.5
set output "../data/rds_nco_out.png"        # Output file name
# set multiplot layout 1,1 scale 1.0,1.0  # set three plots for this figure

# sines
set ylabel 'nco out'                    # set y-axis label
set yrange [-1:1]                       # set y plot range
set xrange [2000:2200]                      # set x plot range
plot '../data/rds_pll_out.dat' using 1:2 with lines lt 1 lw 1 lc rgb '#000088' notitle, \
    '../data/rds_pilot.dat' using 1:($2*10) with lines lt 1 lw 1 lc rgb '#008800' notitle, \
    # '../data/rds_data_apf.dat' using 1:($2*10) with lines lt 1 lw 2 lc rgb '#008800' notitle, \

# uncomment the following block if you want to plot additional data
# set ylabel 'nco out'                      # set y-axis label
# set yrange [-1:1]                     # set y plot range
# set xrange [300:400]                      # set x plot range
# plot '../data/stereo_bpf_filtered.dat' using 1:2 with lines lt 1 lw 2 lc rgb '#000088' notitle, \
#     '../data/stereo_mixed_debug.dat' using 1:2 with lines lt 1 lw 2 lc rgb '#008800' notitle, \
#     '../data/stereo_lpf_filtered_debug.dat' using 1:2 with lines lt 1 lw 2 lc rgb '#880000' notitle, \

# unset multiplot
