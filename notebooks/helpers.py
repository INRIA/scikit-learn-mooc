FIELDNAMES = ('polarity', 'id', 'date', 'query', 'author', 'text')

def read_sentiment_csv(csv_file, fieldnames=FIELDNAMES, max_count=None,
             n_partitions=1, partition_id=0):
    import csv  # put the import inside for use in IPython.parallel
    def file_opener(csv_file):
        try:
            open(csv_file, 'r', encoding="latin1").close()
            return open(csv_file, 'r', encoding="latin1")
        except TypeError:
            # Python 2 does not have encoding arg
            return open(csv_file, 'rb')

    texts = []
    targets = []
    with file_opener(csv_file) as f:
        reader = csv.DictReader(f, fieldnames=fieldnames,
                                delimiter=',', quotechar='"')
        pos_count, neg_count = 0, 0
        for i, d in enumerate(reader):
            if i % n_partitions != partition_id:
                # Skip entry if not in the requested partition
                continue

            if d['polarity'] == '4':
                if max_count and pos_count >= max_count / 2:
                    continue
                pos_count += 1
                texts.append(d['text'])
                targets.append(1)

            elif d['polarity'] == '0':
                if max_count and neg_count >= max_count / 2:
                    continue
                neg_count += 1
                texts.append(d['text'])
                targets.append(-1)

    return texts, targets
