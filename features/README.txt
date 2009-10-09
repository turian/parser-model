Modify hyperparameters.features.yaml, and copy it into the folder in
which you are preprocessing the features.
(Usually, just so later you know what hyperparameters you used.)

./brownclusters.py < infeatures > outfeatures
    Add brown cluster features.

cat infeatures | ./sanitize-colons.pl | ./embeddings.py > outfeatures
    Convert : to COLON and add embeddings features (each feature with a given value).
