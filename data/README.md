# Dataset interchange format

Raw and processed datasets are intentionally excluded from git. Obtain the
six hypergraph-level datasets from the upstream HIC or DHG-Bench repository
after checking their terms, then export one UTF-8 JSON file per dataset:

```json
{
  "samples": [
    {
      "id": "sample-0",
      "label": 0,
      "num_nodes": 4,
      "hyperedges": [[0, 1, 2], [1, 3]],
      "x": [[1.0], [0.0], [0.5], [1.0]]
    }
  ]
}
```

`x` is optional. When absent, the loader creates normalized vertex-degree and
mean-incident-hyperedge-size features. Files belong in `data/processed` using:

- `rhg_3.json`
- `rhg_10.json`
- `imdb_dir_form.json`
- `imdb_dir_genre.json`
- `steam_player.json`
- `twitter_friend.json`
- `highschool.json`
- `primary.json`
- `makam.json`
- `bbc.json`

The loader uses 80/10/10 holdout splits for HIC/DHG-Bench datasets and 10-fold
CV protocol (`--fold 0..9`) for Aktas datasets. It rejects overlapping or
incomplete splits and never deserializes pickle objects. The source repositories are:

- https://github.com/iMoonLab/HIC
- https://github.com/Coco-Hut/DHG-Bench
- https://sites.google.com/view/mehmetaktas/datasets
