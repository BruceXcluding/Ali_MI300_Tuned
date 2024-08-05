for d in */; do
    cd $d
    for f in *.yaml; do
	echo "Tuning for $d/$f"
	afo tune $f --cuda_device 0 1 2 3 4 5 6 7
    done
    cd ..
done
python3 combine_csvs.py
