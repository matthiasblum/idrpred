Due to GitHub's file size limit (100 MiB), the one million sequences FASTA file has been split in smaller files.

```sh
$ split -b 50M --numeric-suffixes=1 input.fasta.gz input.fasta.gz_
```

You can reconstruct the original file with the following command:

```sh
$ cat input.fasta.gz_* > input.fasta.gz
```