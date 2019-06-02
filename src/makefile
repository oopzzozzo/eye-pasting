.PHONY : all clean download_model download_data upload_data

IN=../in/
OUT=../out/

dirs=$(IN) $(OUT)
targets=$(subst _o,_out, $(wildcard $(IN)*_o.jpg))
all: $(dirs) $(subst $(IN), $(OUT), $(targets))

$(dirs):
	mkdir $@
$(OUT)%_out.jpg: $(IN)%_c.jpg $(IN)%_o.jpg
	python3 main.py -b $< -e $(word 2,$^) -o $@

clean:
	rm $(OUT)/*

download_model:
	wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	bunzip2 shape_predictor_68_face_landmarks.dat.bz2 -f

download_data:
	read -p 'CSIE workstation ID: ' uid; \
	rsync $$uid@linux14.csie.org:/tmp2/b05902041/dipFinalData/ ../in/ -r

upload_data: $(IN)
	read -p 'CSIE workstation ID: ' uid; \
	rsync ../in/ $$uid@linux14.csie.org:/tmp2/b05902041/dipFinalData/ -r
