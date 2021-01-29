#!/bin/bash
RESULT_FILE=$1

if [ -f $RESULT_FILE ]; then
    rm $RESULT_FILE
fi
touch $RESULT_FILE

checksum_file() {
    echo `openssl md5 $1 | awk '{print $2}'`
}

FILES=()
while read -r -d ''; do
	  FILES+=("$REPLY")
done < <(find python_scripts -name '*.py' -type f -print0)

# Loop through files and append MD5 to result file
for FILE in ${FILES[@]}; do
	  echo `checksum_file $FILE` >> $RESULT_FILE
done

# Sort the file so that it does not depend on the order of find
sort $RESULT_FILE -o $RESULT_FILE

