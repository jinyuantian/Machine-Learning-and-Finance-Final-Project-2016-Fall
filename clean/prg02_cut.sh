cd ../ZFB-output

cut -d , -f 1,6,7,61-129,131-198,200-274 ZFB-1-NoQuote.csv > ZFB-1-Selected.csv
cut -d , -f 1,6,7,61-129,131-198,200-274 ZFB-2-NoQuote.csv > ZFB-2-Selected.csv
cut -d , -f 1,6,7,61-129,131-198,200-274 ZFB-3-NoQuote.csv > ZFB-3-Selected.csv
cut -d , -f 1,6,7,61-129,131-198,200-274 ZFB-4-NoQuote.csv > ZFB-4-Selected.csv
cut -d , -f 1,6,7,61-129,131-198,200-274 ZFB-5-NoQuote.csv > ZFB-5-Selected.csv

cd ../stg1_clean