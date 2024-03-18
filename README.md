# Introduction

This is a multilingual CCG parser for both CCGBank and multilingual CCG treebanks based on python. Refer to https://github.com/KunhangL/CCG-HPSG-Parser for the older parser (designed specifically for CCGBank) and instructions.

## Installing
1. Create an environment named `ccg`: `conda create -n ccg python=3.10.4`
2. Install packages using `pip`: `pip install -r requirements.txt`
3. If you plan to use CCGBank, please save data (`ccgbank-wsj_00.auto`, `ccgbank-wsj_02-21.auto` and `ccgbank-wsj_23.auto`) under `./data`. If you plan to use data of treebanks, please save all data to `./treebanks` like below.
```
treebanks
├── UD_Afrikaans-AfriBooms
│   ├── af_afribooms-ud-dev.auto
│   ├── af_afribooms-ud-dev.lexicon
│   ├── af_afribooms-ud-test.auto
│   ├── af_afribooms-ud-test.lexicon
│   ├── af_afribooms-ud-train.auto
│   └── af_afribooms-ud-train.lexicon
├── UD_Ancient_Greek-Perseus
│   ├── grc_perseus-ud-dev.auto
│   ├── grc_perseus-ud-dev.lexicon
│   ├── grc_perseus-ud-test.auto
│   ├── grc_perseus-ud-test.lexicon
│   ├── grc_perseus-ud-train.auto
│   └── grc_perseus-ud-train.lexicon
├── ...
...
```
4. Please download and save the corresponding PLM directory under `./plms` like below. (You can only prepare `mT5-base`, which is the default PLM we use in codes.)
```
plms
├── bert-base-multilingual-cased
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── bert-base-uncased
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
└── mt5-base
    ├── config.json
    ├── generation_config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── spiece.model
    └── tokenizer_config.json
```

# Supertagging
The supertagging model is a classifier built upon Transformers. We use the encoder of mT5-base by default.
## Current Best Result
| PLM | Params | Best Result | Multitagging Results on Dev Data (with the best epoch) |
| ----- | ------ | ----------- | ----------------------------- |
| mT5-base | n_epochs = 20;<br>dropout = 0.5;<br>AdamW;<br>lr = 1e-5;<br>batch_size = 10; | best_epoch = 20,<br>train_eval acc =  91.378,<br> dev_eval acc =  86.959 | beta = 0.0005, topk = 10 --> averaged number of lexical categories = 5.03, acc = 97.984;<br><br>beta = 0.0001, topk = 10 --> averaged number of lexical categories = 6.16, acc = 98.123;<br><br>beta = 0.00005, topk = 10 --> averaged number of lexical categories = 6.62, acc = 98.155;<br><br>beta = 0.00001, topk = 10 --> averaged number of lexical categories = 7.51, acc = 98.187;

Evaluation of dev data for each single treebank. Results shown in the order of decreasing training treebank size in terms of n_tokens
```
[UD_German-HDT] || Training size ~ n_sents: 142623, n_tokens: 2475889, ave_token_per_sent: 17.36 || Testing size ~ n_sents: 17192, n_tokens: 286449, ave_token_per_sent: 16.66 || ave_loss: 0.20399958353359685, acc: 94.54492950439453

[UD_Russian-SynTagRus] || Training size ~ n_sents: 64737, n_tokens: 1067352, ave_token_per_sent: 16.49 || Testing size ~ n_sents: 8297, n_tokens: 135217, ave_token_per_sent: 16.3 || ave_loss: 0.36904021011194194, acc: 90.33775329589844

[UD_Czech-PDT] || Training size ~ n_sents: 59481, n_tokens: 956509, ave_token_per_sent: 16.08 || Testing size ~ n_sents: 7979, n_tokens: 129201, ave_token_per_sent: 16.19 || ave_loss: 0.3787599020686589, acc: 89.93041229248047

[UD_Icelandic-IcePaHC] || Training size ~ n_sents: 33083, n_tokens: 665658, ave_token_per_sent: 20.12 || Testing size ~ n_sents: 4687, n_tokens: 127390, ave_token_per_sent: 27.18 || ave_loss: 0.7265830685271383, acc: 81.82823944091797

[UD_Romanian-Nonstandard] || Training size ~ n_sents: 22546, n_tokens: 481614, ave_token_per_sent: 21.36 || Testing size ~ n_sents: 1017, n_tokens: 17712, ave_token_per_sent: 17.42 || ave_loss: 0.6031297251728236, acc: 83.96002960205078

[UD_Spanish-AnCora] || Training size ~ n_sents: 13360, n_tokens: 405954, ave_token_per_sent: 30.39 || Testing size ~ n_sents: 1519, n_tokens: 46797, ave_token_per_sent: 30.81 || ave_loss: 0.3458706690488677, acc: 91.57424926757812

[UD_Czech-CAC] || Training size ~ n_sents: 20419, n_tokens: 388531, ave_token_per_sent: 19.03 || Testing size ~ n_sents: 551, n_tokens: 9689, ave_token_per_sent: 17.58 || ave_loss: 1.9561946247039097, acc: 78.97615051269531

[UD_Persian-PerDT] || Training size ~ n_sents: 22445, n_tokens: 358035, ave_token_per_sent: 15.95 || Testing size ~ n_sents: 1246, n_tokens: 19678, ave_token_per_sent: 15.79 || ave_loss: 0.3706875228881836, acc: 90.41569519042969

[UD_Catalan-AnCora] || Training size ~ n_sents: 11254, n_tokens: 343074, ave_token_per_sent: 30.48 || Testing size ~ n_sents: 1455, n_tokens: 45785, ave_token_per_sent: 31.47 || ave_loss: 0.31733518139752626, acc: 92.33592224121094

[UD_Spanish-GSD] || Training size ~ n_sents: 12969, n_tokens: 334792, ave_token_per_sent: 25.81 || Testing size ~ n_sents: 1268, n_tokens: 32327, ave_token_per_sent: 25.49 || ave_loss: 0.4872328211942057, acc: 87.35113525390625

[UD_French-GSD] || Training size ~ n_sents: 13601, n_tokens: 322274, ave_token_per_sent: 23.69 || Testing size ~ n_sents: 1388, n_tokens: 32775, ave_token_per_sent: 23.61 || ave_loss: 0.2557284160912466, acc: 92.91227722167969

[UD_Estonian-EDT] || Training size ~ n_sents: 23232, n_tokens: 312121, ave_token_per_sent: 13.43 || Testing size ~ n_sents: 2843, n_tokens: 39383, ave_token_per_sent: 13.85 || ave_loss: 0.572241225080532, acc: 86.12599182128906

[UD_Italian-ISDT] || Training size ~ n_sents: 12747, n_tokens: 258514, ave_token_per_sent: 20.28 || Testing size ~ n_sents: 542, n_tokens: 10990, ave_token_per_sent: 20.28 || ave_loss: 0.31557465249841865, acc: 91.24658966064453

[UD_Polish-PDB] || Training size ~ n_sents: 16167, n_tokens: 245243, ave_token_per_sent: 15.17 || Testing size ~ n_sents: 2013, n_tokens: 30271, ave_token_per_sent: 15.04 || ave_loss: 0.3909931227029993, acc: 88.55339813232422

[UD_Belarusian-HSE] || Training size ~ n_sents: 21011, n_tokens: 235969, ave_token_per_sent: 11.23 || Testing size ~ n_sents: 1175, n_tokens: 13056, ave_token_per_sent: 11.11 || ave_loss: 0.40820041889230074, acc: 89.82843017578125

[UD_Classical_Chinese-Kyoto] || Training size ~ n_sents: 47905, n_tokens: 231433, ave_token_per_sent: 4.83 || Testing size ~ n_sents: 5520, n_tokens: 27179, ave_token_per_sent: 4.92 || ave_loss: 0.7152479682758829, acc: 79.96614837646484

[UD_Hindi-HDTB] || Training size ~ n_sents: 11062, n_tokens: 219991, ave_token_per_sent: 19.89 || Testing size ~ n_sents: 1400, n_tokens: 28176, ave_token_per_sent: 20.13 || ave_loss: 0.41201430011008466, acc: 88.63216400146484

[UD_Norwegian-Bokmaal] || Training size ~ n_sents: 14449, n_tokens: 215263, ave_token_per_sent: 14.9 || Testing size ~ n_sents: 2229, n_tokens: 32216, ave_token_per_sent: 14.45 || ave_loss: 0.22726076677224427, acc: 94.0495376586914

[UD_Norwegian-Nynorsk] || Training size ~ n_sents: 12972, n_tokens: 214325, ave_token_per_sent: 16.52 || Testing size ~ n_sents: 1738, n_tokens: 27584, ave_token_per_sent: 15.87 || ave_loss: 0.30332234258838425, acc: 92.50652313232422

[UD_German-GSD] || Training size ~ n_sents: 11888, n_tokens: 212175, ave_token_per_sent: 17.85 || Testing size ~ n_sents: 648, n_tokens: 9292, ave_token_per_sent: 14.34 || ave_loss: 0.6013183150153893, acc: 86.24623107910156

[UD_Korean-Kaist] || Training size ~ n_sents: 17574, n_tokens: 211493, ave_token_per_sent: 12.03 || Testing size ~ n_sents: 1576, n_tokens: 17836, ave_token_per_sent: 11.32 || ave_loss: 0.6054604967372327, acc: 84.58735656738281

[UD_Portuguese-GSD] || Training size ~ n_sents: 8383, n_tokens: 208941, ave_token_per_sent: 24.92 || Testing size ~ n_sents: 1033, n_tokens: 26019, ave_token_per_sent: 25.19 || ave_loss: 0.26256496946399027, acc: 92.45935821533203

[UD_Latin-ITTB] || Training size ~ n_sents: 14400, n_tokens: 206338, ave_token_per_sent: 14.33 || Testing size ~ n_sents: 1380, n_tokens: 16089, ave_token_per_sent: 11.66 || ave_loss: 0.4428451117128134, acc: 88.94896697998047

[UD_Italian-VIT] || Training size ~ n_sents: 7568, n_tokens: 190665, ave_token_per_sent: 25.19 || Testing size ~ n_sents: 681, n_tokens: 24516, ave_token_per_sent: 36.0 || ave_loss: 0.3891491754979327, acc: 89.7740249633789

[UD_English-EWT] || Training size ~ n_sents: 11254, n_tokens: 173967, ave_token_per_sent: 15.46 || Testing size ~ n_sents: 1835, n_tokens: 21945, ave_token_per_sent: 11.96 || ave_loss: 0.4353062829164707, acc: 87.96270751953125

[UD_Latvian-LVTB] || Training size ~ n_sents: 10614, n_tokens: 170801, ave_token_per_sent: 16.09 || Testing size ~ n_sents: 1675, n_tokens: 24793, ave_token_per_sent: 14.8 || ave_loss: 0.5098282019164235, acc: 86.78659057617188

[UD_Arabic-PADT] || Training size ~ n_sents: 4975, n_tokens: 159423, ave_token_per_sent: 32.04 || Testing size ~ n_sents: 698, n_tokens: 21296, ave_token_per_sent: 30.51 || ave_loss: 0.7306396612099239, acc: 82.31123352050781

[UD_Japanese-GSD] || Training size ~ n_sents: 6794, n_tokens: 158851, ave_token_per_sent: 23.38 || Testing size ~ n_sents: 489, n_tokens: 11706, ave_token_per_sent: 23.94 || ave_loss: 0.38452432091747013, acc: 90.25286102294922

[UD_Romanian-RRT] || Training size ~ n_sents: 6809, n_tokens: 150342, ave_token_per_sent: 22.08 || Testing size ~ n_sents: 626, n_tokens: 13547, ave_token_per_sent: 21.64 || ave_loss: 0.48707828966398087, acc: 88.25569915771484

[UD_Turkish-Penn] || Training size ~ n_sents: 13525, n_tokens: 147539, ave_token_per_sent: 10.91 || Testing size ~ n_sents: 576, n_tokens: 6277, ave_token_per_sent: 10.9 || ave_loss: 0.7424036254142893, acc: 78.3335952758789

[UD_Russian-Taiga] || Training size ~ n_sents: 14449, n_tokens: 145755, ave_token_per_sent: 10.09 || Testing size ~ n_sents: 809, n_tokens: 7858, ave_token_per_sent: 9.71 || ave_loss: 1.0193609206212892, acc: 75.74446105957031

[UD_Dutch-Alpino] || Training size ~ n_sents: 10425, n_tokens: 143975, ave_token_per_sent: 13.81 || Testing size ~ n_sents: 630, n_tokens: 9457, ave_token_per_sent: 15.01 || ave_loss: 0.38703948566837915, acc: 89.4575424194336

[UD_Finnish-TDT] || Training size ~ n_sents: 10954, n_tokens: 139746, ave_token_per_sent: 12.76 || Testing size ~ n_sents: 1237, n_tokens: 15688, ave_token_per_sent: 12.68 || ave_loss: 0.530244565869291, acc: 85.82994079589844

[UD_Portuguese-Bosque] || Training size ~ n_sents: 6014, n_tokens: 138083, ave_token_per_sent: 22.96 || Testing size ~ n_sents: 1007, n_tokens: 22775, ave_token_per_sent: 22.62 || ave_loss: 0.3896965224406507, acc: 89.88803100585938

[UD_Turkish-Kenet] || Training size ~ n_sents: 14663, n_tokens: 134272, ave_token_per_sent: 9.16 || Testing size ~ n_sents: 1562, n_tokens: 16414, ave_token_per_sent: 10.51 || ave_loss: 1.0017769503745304, acc: 74.0770034790039

[UD_Croatian-SET] || Training size ~ n_sents: 6174, n_tokens: 130442, ave_token_per_sent: 21.13 || Testing size ~ n_sents: 849, n_tokens: 18858, ave_token_per_sent: 22.21 || ave_loss: 0.4803919949952294, acc: 88.18537902832031

[UD_Hebrew-HTB] || Training size ~ n_sents: 5021, n_tokens: 127476, ave_token_per_sent: 25.39 || Testing size ~ n_sents: 472, n_tokens: 10802, ave_token_per_sent: 22.89 || ave_loss: 0.4419611959407727, acc: 89.69635009765625

[UD_Japanese-GSDLUW] || Training size ~ n_sents: 6816, n_tokens: 123653, ave_token_per_sent: 18.14 || Testing size ~ n_sents: 489, n_tokens: 9100, ave_token_per_sent: 18.61 || ave_loss: 0.2910939544743421, acc: 92.56044006347656

[UD_Old_French-SRCMF] || Training size ~ n_sents: 11874, n_tokens: 118316, ave_token_per_sent: 9.96 || Testing size ~ n_sents: 1575, n_tokens: 15131, ave_token_per_sent: 9.61 || ave_loss: 0.5612638631387602, acc: 85.61893463134766

[UD_Bulgarian-BTB] || Training size ~ n_sents: 8615, n_tokens: 117513, ave_token_per_sent: 13.64 || Testing size ~ n_sents: 1086, n_tokens: 15275, ave_token_per_sent: 14.07 || ave_loss: 1.1381425332585606, acc: 81.23731994628906

[UD_Icelandic-Modern] || Training size ~ n_sents: 5130, n_tokens: 115833, ave_token_per_sent: 22.58 || Testing size ~ n_sents: 742, n_tokens: 15775, ave_token_per_sent: 21.26 || ave_loss: 0.5410747796297073, acc: 87.10617065429688

[UD_Finnish-FTB] || Training size ~ n_sents: 13778, n_tokens: 111627, ave_token_per_sent: 8.1 || Testing size ~ n_sents: 1716, n_tokens: 13500, ave_token_per_sent: 7.87 || ave_loss: 0.5635133101279999, acc: 84.7851791381836

[UD_Czech-FicTree] || Training size ~ n_sents: 9002, n_tokens: 106830, ave_token_per_sent: 11.87 || Testing size ~ n_sents: 1162, n_tokens: 13451, ave_token_per_sent: 11.58 || ave_loss: 0.3540819593283356, acc: 89.74053955078125

[UD_Persian-Seraji] || Training size ~ n_sents: 4462, n_tokens: 105970, ave_token_per_sent: 23.75 || Testing size ~ n_sents: 560, n_tokens: 14045, ave_token_per_sent: 25.08 || ave_loss: 0.6739260980061123, acc: 81.77999114990234

[UD_Polish-LFG] || Training size ~ n_sents: 13667, n_tokens: 103604, ave_token_per_sent: 7.58 || Testing size ~ n_sents: 1739, n_tokens: 13024, ave_token_per_sent: 7.49 || ave_loss: 0.37631598101051034, acc: 88.52118682861328

[UD_Latin-PROIEL] || Training size ~ n_sents: 11443, n_tokens: 94757, ave_token_per_sent: 8.28 || Testing size ~ n_sents: 847, n_tokens: 7335, ave_token_per_sent: 8.66 || ave_loss: 0.7682609861387926, acc: 79.91819763183594

[UD_Slovenian-SSJ] || Training size ~ n_sents: 5689, n_tokens: 94561, ave_token_per_sent: 16.62 || Testing size ~ n_sents: 617, n_tokens: 11400, ave_token_per_sent: 18.48 || ave_loss: 0.30855319168298473, acc: 92.41228485107422

[UD_Ancient_Greek-PROIEL] || Training size ~ n_sents: 9399, n_tokens: 91901, ave_token_per_sent: 9.78 || Testing size ~ n_sents: 612, n_tokens: 6579, ave_token_per_sent: 10.75 || ave_loss: 2.6672701104994743, acc: 65.45067596435547

[UD_Old_East_Slavic-TOROT] || Training size ~ n_sents: 11317, n_tokens: 90658, ave_token_per_sent: 8.01 || Testing size ~ n_sents: 1574, n_tokens: 12032, ave_token_per_sent: 7.64 || ave_loss: 0.8016182303051406, acc: 78.85637664794922

[UD_Chinese-GSD] || Training size ~ n_sents: 3728, n_tokens: 90199, ave_token_per_sent: 24.2 || Testing size ~ n_sents: 474, n_tokens: 11773, ave_token_per_sent: 24.84 || ave_loss: 0.6287098446240028, acc: 85.22042083740234

[UD_English-GUM] || Training size ~ n_sents: 5096, n_tokens: 89355, ave_token_per_sent: 17.53 || Testing size ~ n_sents: 784, n_tokens: 13891, ave_token_per_sent: 17.72 || ave_loss: 0.400859921510461, acc: 88.80570220947266

[UD_Turkish-BOUN] || Training size ~ n_sents: 7244, n_tokens: 86869, ave_token_per_sent: 11.99 || Testing size ~ n_sents: 909, n_tokens: 10733, ave_token_per_sent: 11.81 || ave_loss: 1.1756272463353126, acc: 69.05804443359375

[UD_Indonesian-GSD] || Training size ~ n_sents: 4138, n_tokens: 85423, ave_token_per_sent: 20.64 || Testing size ~ n_sents: 503, n_tokens: 10761, ave_token_per_sent: 21.39 || ave_loss: 0.7579179996368932, acc: 80.03903198242188

[UD_Italian-PoSTWITA] || Training size ~ n_sents: 4646, n_tokens: 84682, ave_token_per_sent: 18.23 || Testing size ~ n_sents: 563, n_tokens: 10087, ave_token_per_sent: 17.92 || ave_loss: 0.7505603343771216, acc: 81.51085662841797

[UD_Chinese-GSDSimp] || Training size ~ n_sents: 3544, n_tokens: 83567, ave_token_per_sent: 23.58 || Testing size ~ n_sents: 444, n_tokens: 10831, ave_token_per_sent: 24.39 || ave_loss: 0.5980112575822406, acc: 85.6984634399414

[UD_Romanian-SiMoNERo] || Training size ~ n_sents: 2908, n_tokens: 81363, ave_token_per_sent: 27.98 || Testing size ~ n_sents: 348, n_tokens: 10675, ave_token_per_sent: 30.68 || ave_loss: 1.1469638807432991, acc: 81.42388153076172

[UD_Ukrainian-IU] || Training size ~ n_sents: 4901, n_tokens: 76928, ave_token_per_sent: 15.7 || Testing size ~ n_sents: 590, n_tokens: 10234, ave_token_per_sent: 17.35 || ave_loss: 0.4767779902381412, acc: 87.42427062988281

[UD_Galician-CTG] || Training size ~ n_sents: 2210, n_tokens: 76573, ave_token_per_sent: 34.65 || Testing size ~ n_sents: 846, n_tokens: 29115, ave_token_per_sent: 34.41 || ave_loss: 1.1364294052124024, acc: 75.31856536865234

[UD_Urdu-UDTB] || Training size ~ n_sents: 3077, n_tokens: 74832, ave_token_per_sent: 24.32 || Testing size ~ n_sents: 424, n_tokens: 10160, ave_token_per_sent: 23.96 || ave_loss: 0.7032842001942701, acc: 81.21063232421875

[UD_Turkish-Tourism] || Training size ~ n_sents: 15277, n_tokens: 70112, ave_token_per_sent: 4.59 || Testing size ~ n_sents: 2162, n_tokens: 10277, ave_token_per_sent: 4.75 || ave_loss: 0.3322122536979461, acc: 89.81220245361328

[UD_Irish-IDT] || Training size ~ n_sents: 3032, n_tokens: 66316, ave_token_per_sent: 21.87 || Testing size ~ n_sents: 316, n_tokens: 6011, ave_token_per_sent: 19.02 || ave_loss: 0.973869888111949, acc: 78.30643463134766

[UD_Serbian-SET] || Training size ~ n_sents: 3083, n_tokens: 65593, ave_token_per_sent: 21.28 || Testing size ~ n_sents: 486, n_tokens: 10243, ave_token_per_sent: 21.08 || ave_loss: 0.3817359300292268, acc: 91.07683563232422

[UD_Dutch-LassySmall] || Training size ~ n_sents: 5320, n_tokens: 64301, ave_token_per_sent: 12.09 || Testing size ~ n_sents: 605, n_tokens: 9151, ave_token_per_sent: 15.13 || ave_loss: 0.47722327010706067, acc: 86.41677856445312

[UD_Slovak-SNK] || Training size ~ n_sents: 7008, n_tokens: 64251, ave_token_per_sent: 9.17 || Testing size ~ n_sents: 971, n_tokens: 11209, ave_token_per_sent: 11.54 || ave_loss: 0.34890312445825156, acc: 91.12320709228516

[UD_Western_Armenian-ArmTDP] || Training size ~ n_sents: 3627, n_tokens: 63114, ave_token_per_sent: 17.4 || Testing size ~ n_sents: 437, n_tokens: 7654, ave_token_per_sent: 17.51 || ave_loss: 0.7502026706933975, acc: 80.74209594726562

[UD_Swedish-Talbanken] || Training size ~ n_sents: 3984, n_tokens: 58673, ave_token_per_sent: 14.73 || Testing size ~ n_sents: 444, n_tokens: 8123, ave_token_per_sent: 18.3 || ave_loss: 0.45270336733924016, acc: 88.18170166015625

[UD_Danish-DDT] || Training size ~ n_sents: 3428, n_tokens: 53991, ave_token_per_sent: 15.75 || Testing size ~ n_sents: 441, n_tokens: 7005, ave_token_per_sent: 15.88 || ave_loss: 0.5173318127791087, acc: 87.59457397460938

[UD_Scottish_Gaelic-ARCOSG] || Training size ~ n_sents: 2824, n_tokens: 51062, ave_token_per_sent: 18.08 || Testing size ~ n_sents: 576, n_tokens: 8174, ave_token_per_sent: 14.19 || ave_loss: 0.9974150678207134, acc: 77.75875091552734

[UD_Swedish-LinES] || Training size ~ n_sents: 2994, n_tokens: 49936, ave_token_per_sent: 16.68 || Testing size ~ n_sents: 957, n_tokens: 16173, ave_token_per_sent: 16.9 || ave_loss: 0.5265347788420817, acc: 87.1205062866211

[UD_English-LinES] || Training size ~ n_sents: 2833, n_tokens: 49176, ave_token_per_sent: 17.36 || Testing size ~ n_sents: 898, n_tokens: 15662, ave_token_per_sent: 17.44 || ave_loss: 0.5170677006244659, acc: 86.34912109375

[UD_English-Atis] || Training size ~ n_sents: 4193, n_tokens: 47617, ave_token_per_sent: 11.36 || Testing size ~ n_sents: 554, n_tokens: 6414, ave_token_per_sent: 11.58 || ave_loss: 0.28618574192764107, acc: 92.89054870605469

[UD_French-Sequoia] || Training size ~ n_sents: 2153, n_tokens: 46955, ave_token_per_sent: 21.81 || Testing size ~ n_sents: 394, n_tokens: 9156, ave_token_per_sent: 23.24 || ave_loss: 0.35191274154931307, acc: 90.09392547607422

[UD_Latin-LLCT] || Training size ~ n_sents: 1802, n_tokens: 45118, ave_token_per_sent: 25.04 || Testing size ~ n_sents: 241, n_tokens: 6194, ave_token_per_sent: 25.7 || ave_loss: 0.36391018378490114, acc: 92.05683135986328

[UD_Italian-ParTUT] || Training size ~ n_sents: 1708, n_tokens: 44976, ave_token_per_sent: 26.33 || Testing size ~ n_sents: 151, n_tokens: 2795, ave_token_per_sent: 18.51 || ave_loss: 0.37600249610841274, acc: 90.76922607421875

[UD_Estonian-EWT] || Training size ~ n_sents: 3756, n_tokens: 42832, ave_token_per_sent: 11.4 || Testing size ~ n_sents: 747, n_tokens: 8593, ave_token_per_sent: 11.5 || ave_loss: 0.5895925744374593, acc: 85.12742614746094

[UD_Basque-BDT] || Training size ~ n_sents: 3677, n_tokens: 42361, ave_token_per_sent: 11.52 || Testing size ~ n_sents: 1223, n_tokens: 13663, ave_token_per_sent: 11.17 || ave_loss: 0.6827819407955418, acc: 82.95396423339844

[UD_Russian-GSD] || Training size ~ n_sents: 2497, n_tokens: 42223, ave_token_per_sent: 16.91 || Testing size ~ n_sents: 370, n_tokens: 6353, ave_token_per_sent: 17.17 || ave_loss: 0.6850266198854189, acc: 83.00016021728516

[UD_English-ParTUT] || Training size ~ n_sents: 1707, n_tokens: 40136, ave_token_per_sent: 23.51 || Testing size ~ n_sents: 151, n_tokens: 2580, ave_token_per_sent: 17.09 || ave_loss: 0.5216125813312829, acc: 87.24806213378906

[UD_Greek-GDT] || Training size ~ n_sents: 1543, n_tokens: 37999, ave_token_per_sent: 24.63 || Testing size ~ n_sents: 376, n_tokens: 9513, ave_token_per_sent: 25.3 || ave_loss: 0.7168417451413054, acc: 84.82077026367188

[UD_Korean-GSD] || Training size ~ n_sents: 3465, n_tokens: 36747, ave_token_per_sent: 10.61 || Testing size ~ n_sents: 750, n_tokens: 7773, ave_token_per_sent: 10.36 || ave_loss: 1.3697897024949393, acc: 67.83738708496094

[UD_Turkish-Atis] || Training size ~ n_sents: 4225, n_tokens: 35586, ave_token_per_sent: 8.42 || Testing size ~ n_sents: 567, n_tokens: 4793, ave_token_per_sent: 8.45 || ave_loss: 0.559838297531793, acc: 86.22991180419922

[UD_Lithuanian-ALKSNIS] || Training size ~ n_sents: 1890, n_tokens: 33866, ave_token_per_sent: 17.92 || Testing size ~ n_sents: 436, n_tokens: 7353, ave_token_per_sent: 16.86 || ave_loss: 0.7538191428916021, acc: 81.58574676513672

[UD_Armenian-ArmTDP] || Training size ~ n_sents: 1704, n_tokens: 33051, ave_token_per_sent: 19.4 || Testing size ~ n_sents: 218, n_tokens: 4248, ave_token_per_sent: 19.49 || ave_loss: 0.701201053505594, acc: 79.73163604736328

[UD_Turkish-IMST] || Training size ~ n_sents: 3404, n_tokens: 32776, ave_token_per_sent: 9.63 || Testing size ~ n_sents: 904, n_tokens: 8536, ave_token_per_sent: 9.44 || ave_loss: 1.3688426391109005, acc: 62.81631088256836

[UD_Afrikaans-AfriBooms] || Training size ~ n_sents: 1195, n_tokens: 28866, ave_token_per_sent: 24.16 || Testing size ~ n_sents: 185, n_tokens: 4820, ave_token_per_sent: 26.05 || ave_loss: 0.855294040943447, acc: 79.83402252197266

[UD_Old_Church_Slavonic-PROIEL] || Training size ~ n_sents: 3448, n_tokens: 28049, ave_token_per_sent: 8.13 || Testing size ~ n_sents: 897, n_tokens: 7382, ave_token_per_sent: 8.23 || ave_loss: 0.845256151093377, acc: 78.42047882080078

[UD_Gothic-PROIEL] || Training size ~ n_sents: 2819, n_tokens: 25256, ave_token_per_sent: 8.96 || Testing size ~ n_sents: 797, n_tokens: 7082, ave_token_per_sent: 8.89 || ave_loss: 1.1171187922358512, acc: 70.95453643798828

[UD_Ancient_Greek-Perseus] || Training size ~ n_sents: 2347, n_tokens: 22816, ave_token_per_sent: 9.72 || Testing size ~ n_sents: 206, n_tokens: 2880, ave_token_per_sent: 13.98 || ave_loss: 1.0867802983238584, acc: 72.11805725097656

[UD_Wolof-WTB] || Training size ~ n_sents: 1145, n_tokens: 22129, ave_token_per_sent: 19.33 || Testing size ~ n_sents: 426, n_tokens: 9315, ave_token_per_sent: 21.87 || ave_loss: 1.245098349659942, acc: 69.74771881103516

[UD_Faroese-FarPaHC] || Training size ~ n_sents: 1014, n_tokens: 22097, ave_token_per_sent: 21.79 || Testing size ~ n_sents: 296, n_tokens: 8310, ave_token_per_sent: 28.07 || ave_loss: 0.6107086320718129, acc: 85.10228729248047

[UD_Maltese-MUDT] || Training size ~ n_sents: 1072, n_tokens: 21002, ave_token_per_sent: 19.59 || Testing size ~ n_sents: 393, n_tokens: 8950, ave_token_per_sent: 22.77 || ave_loss: 2.7809042781591415, acc: 66.43575286865234

[UD_French-ParTUT] || Training size ~ n_sents: 717, n_tokens: 19722, ave_token_per_sent: 27.51 || Testing size ~ n_sents: 104, n_tokens: 1755, ave_token_per_sent: 16.88 || ave_loss: 0.43172089891000226, acc: 88.8888931274414

[UD_Italian-TWITTIRO] || Training size ~ n_sents: 951, n_tokens: 19321, ave_token_per_sent: 20.32 || Testing size ~ n_sents: 123, n_tokens: 2481, ave_token_per_sent: 20.17 || ave_loss: 0.7226600096775935, acc: 82.5070571899414

[UD_Vietnamese-VTB] || Training size ~ n_sents: 1336, n_tokens: 18680, ave_token_per_sent: 13.98 || Testing size ~ n_sents: 769, n_tokens: 10726, ave_token_per_sent: 13.95 || ave_loss: 1.447933057685951, acc: 66.12903594970703

[UD_Turkish-FrameNet] || Training size ~ n_sents: 2281, n_tokens: 16268, ave_token_per_sent: 7.13 || Testing size ~ n_sents: 204, n_tokens: 1406, ave_token_per_sent: 6.89 || ave_loss: 0.7962737267925626, acc: 77.24039459228516

[UD_Uyghur-UDT] || Training size ~ n_sents: 1449, n_tokens: 16019, ave_token_per_sent: 11.06 || Testing size ~ n_sents: 801, n_tokens: 8869, ave_token_per_sent: 11.07 || ave_loss: 1.2262895026324707, acc: 66.7042465209961

[UD_French-Rhapsodie] || Training size ~ n_sents: 1185, n_tokens: 16006, ave_token_per_sent: 13.51 || Testing size ~ n_sents: 1008, n_tokens: 11036, ave_token_per_sent: 10.95 || ave_loss: 0.5652795010569072, acc: 85.04893493652344

[UD_Hungarian-Szeged] || Training size ~ n_sents: 732, n_tokens: 14869, ave_token_per_sent: 20.31 || Testing size ~ n_sents: 320, n_tokens: 7412, ave_token_per_sent: 23.16 || ave_loss: 0.6865596557036042, acc: 83.59416961669922

[UD_Welsh-CCG] || Training size ~ n_sents: 731, n_tokens: 14493, ave_token_per_sent: 19.83 || Testing size ~ n_sents: 402, n_tokens: 8782, ave_token_per_sent: 21.85 || ave_loss: 0.7600899207882765, acc: 81.31404876708984

[UD_Latin-UDante] || Training size ~ n_sents: 480, n_tokens: 13552, ave_token_per_sent: 28.23 || Testing size ~ n_sents: 170, n_tokens: 4225, ave_token_per_sent: 24.85 || ave_loss: 1.0932055816930883, acc: 73.5621337890625

[UD_Norwegian-NynorskLIA] || Training size ~ n_sents: 1866, n_tokens: 10722, ave_token_per_sent: 5.75 || Testing size ~ n_sents: 388, n_tokens: 2051, ave_token_per_sent: 5.29 || ave_loss: 0.6516136208978983, acc: 83.61774444580078

[UD_Coptic-Scriptorium] || Training size ~ n_sents: 335, n_tokens: 9173, ave_token_per_sent: 27.38 || Testing size ~ n_sents: 173, n_tokens: 4771, ave_token_per_sent: 27.58 || ave_loss: 2.308559331629011, acc: 50.995601654052734

[UD_Turkish_German-SAGT] || Training size ~ n_sents: 503, n_tokens: 7973, ave_token_per_sent: 15.85 || Testing size ~ n_sents: 683, n_tokens: 10237, ave_token_per_sent: 14.99 || ave_loss: 1.4520702638487886, acc: 65.7614517211914

[UD_Czech-CLTT] || Training size ~ n_sents: 421, n_tokens: 6495, ave_token_per_sent: 15.43 || Testing size ~ n_sents: 49, n_tokens: 769, ave_token_per_sent: 15.69 || ave_loss: 0.5334645748138428, acc: 87.64629364013672

[UD_Tamil-TTB] || Training size ~ n_sents: 385, n_tokens: 6045, ave_token_per_sent: 15.7 || Testing size ~ n_sents: 80, n_tokens: 1263, ave_token_per_sent: 15.79 || ave_loss: 1.2758336886763573, acc: 65.71654510498047

[UD_Telugu-MTG] || Training size ~ n_sents: 1050, n_tokens: 5073, ave_token_per_sent: 4.83 || Testing size ~ n_sents: 131, n_tokens: 662, ave_token_per_sent: 5.05 || ave_loss: 0.6876610810203212, acc: 82.17522430419922

[UD_Marathi-UFAL] || Training size ~ n_sents: 357, n_tokens: 2795, ave_token_per_sent: 7.83 || Testing size ~ n_sents: 42, n_tokens: 366, ave_token_per_sent: 8.71 || ave_loss: 1.0074745655059814, acc: 73.22404479980469

[UD_Lithuanian-HSE] || Training size ~ n_sents: 108, n_tokens: 2103, ave_token_per_sent: 19.47 || Testing size ~ n_sents: 41, n_tokens: 796, ave_token_per_sent: 19.41 || ave_loss: 0.9869348645210266, acc: 73.24120330810547

[UD_Swedish_Sign_Language-SSLC] || Training size ~ n_sents: 63, n_tokens: 350, ave_token_per_sent: 5.56 || Testing size ~ n_sents: 48, n_tokens: 311, ave_token_per_sent: 6.48 || ave_loss: 2.6652204513549806, acc: 31.511253356933594
```

## Train a Supertagging Model
```
cd ccg_supertagger
bash run_trainer.sh
```
**NOTE 1**: Using `AutoTokenizer` in `trainer.py` and `supertagger.py`, and `MT5EncoderModel` in `models.py`. Needed to switch to `BertModel` in `models.py` when using BERT-family models.<br>
**NOTE 2**: Please remember to update `--treebanks_train_data_paths`, `--treebanks_dev_data_paths` and `--treebanks_test_data_paths` when updating the data of treebanks.<br>
**NOTE 3**: Please specify different parameters in `run_trainer.sh` so as to use different functions.<br>

### Important Parameters
`--lexical_category2idx_path`: The relative path to the dictionary mapping each CCG category to its index. Please specify either `../data/lexical_category2idx_cutoff_ccgbank.json` or `../data/lexical_category2idx_cutoff_treebanks.json`.
- If you want to build your new category2idx dictionary for new treebanks data, please go to `./multiCCG` and run the `build_category2idx` function in `tools.py`, e.g., `build_category2idx(folder_path='./treebanks', result_path='./data/treebanks_lexical_category2idx_cutoff.json')`.

`--model_dir`: The relative path to the directory of the pretrained language model. Default to `../plms/mt5-base`.<br>
`--checkpoints_dir`: The directory created under `ccg_supertagger` to save checkpoints.<br>
`--n_epochs`: The total number of epochs, default to be `20`.<br>
`--device`: The device to run the experiment, default to be `cuda:0`.<br>
`--batch_size`: Default to be `8`.<br>
`--lr`: Default to be `1e-5`.<br>
`--dropout_p`: The probability of dropout, default to be `0.5`.<br>
`--mode`: The mode to use the trainer. Choices include `train`, `train_on` and `test`, default to `train`. `train` is for training from scratch. `train_on` is for training from a specific checkpoint, in which case `--checkpoint_epoch` should be specified. `test` is for testing on one dataset using the model from a specific checkpoint, in which case `--checkpoint_epoch` and `--test_mode` should be specified.<br>
`--test_mode`: Only specified when it is `test` mode. Choices include `train_eval`, `dev_eval` and `test_eval`, default to `dev_eval`.<br>
`--checkpoint_epoch`: Only for `train_on` and `test` mode, the specific epoch of checkpoint to use.<br>
`--data_mode`: This specifies which dataset to use. Choices include `ccgbank` and `treebanks`.<br>
`--load_mode`: If running the first script on a certain dataset (`ccgbank` or `treebanks`), please specify `first`, in which case the codes will build all necessary datasets from raw data, and store them as `.pkl` files under `ccg_supertagger`. Note that it takes long (one hour~) to prepare data of treebanks for the first time. Afterwards, please always specify `reuse` so as to directly load these saved files.<br>

### Some Example Scripts

- Training on **CCGBank** from scratch
```
#!/bin/bash

DATA="ccgbank"
python -u trainer.py \
 --batch_size 10 \
 --dropout_p 0.5 \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_ccgbank.json \
 --model_dir ../plms/mt5-base \
 --checkpoints_dir ./checkpoints_$DATA \
 --mode train \
 --n_epochs 20 \
 --data_mode ccgbank \
 --load_mode first \
 2>&1 | tee -a supertagging_train_$DATA.log
```

- Training on **treebanks** from scratch
  - NOTE: When training for the first time, if CUDA returns `out of memory` during the first epoch, it is better to set the batch size to be 1 and train for one epoch, so as to filter out ids of problematic data which will be printed out when running. E.g., `('../treebanks/UD_Belarusian-HSE/be_hse-ud-train.auto_ID=radyjosvaboda-6682 PARSER=GOLD NUMPARSE=1',)`. Remember to collect such data entries and store them in `problemantic_ids.txt` (Check this file for how the entries should be kept.)
```
#!/bin/bash

DATA="treebanks"
python -u trainer.py \
 --batch_size 10 \
 --dropout_p 0.5 \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_treebanks.json \
 --model_dir ../plms/mt5-base \
 --checkpoints_dir ./checkpoints_$DATA \
 --mode train \
 --n_epochs 20 \
 --data_mode treebanks \
 --load_mode first \
 2>&1 | tee -a supertagging_train_$DATA.log
```

- Continuing to train the supertagging model on **CCGBank** from checkpoint of epoch 3
```
#!/bin/bash

DATA="ccgbank"
python -u trainer.py \
 --batch_size 10 \
 --dropout_p 0.5 \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_ccgbank.json \
 --model_dir ../plms/mt5-base \
 --checkpoints_dir ./checkpoints_$DATA \
 --mode train_on \
 --n_epochs 20 \
 --checkpoint_epoch 3 \
 --data_mode ccgbank \
 --load_mode reuse \
 2>&1 | tee -a supertagging_train_$DATA.log
```

- Continuing to train the supertagging model on **treebanks** from checkpoint of epoch 3
```
#!/bin/bash

DATA="treebanks"
python -u trainer.py \
 --batch_size 10 \
 --dropout_p 0.5 \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_treebanks.json \
 --model_dir ../plms/mt5-base \
 --checkpoints_dir ./checkpoints_$DATA \
 --mode train_on \
 --n_epochs 20 \
 --checkpoint_epoch 3 \
 --data_mode treebanks \
 --load_mode reuse \
 2>&1 | tee -a supertagging_train_$DATA.log
```

- Testing the supertagging model on dev data of **CCGBank** using the checkpoint from epoch 8
```
#!/bin/bash

DATA="ccgbank"
python -u trainer.py \
 --batch_size 10 \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_ccgbank.json \
 --model_dir ../plms/mt5-base \
 --checkpoints_dir ./checkpoints_$DATA \
 --mode test \
 --test_mode dev_eval \
 --checkpoint_epoch 8 \
 --data_mode ccgbank \
 --load_mode reuse \
 2>&1 | tee -a supertagging_test_$DATA.log
```

- Testing the supertagging model on dev data of **treebanks** using the checkpoint from epoch 20 (different from CCGBank, loading checkpoints and testing on treebank data will return sorted evaluation results for each treebank)
```
#!/bin/bash

DATA="treebanks"
python -u trainer.py \
 --batch_size 10 \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_treebanks.json \
 --model_dir ../plms/mt5-base \
 --checkpoints_dir ./checkpoints_$DATA \
 --mode test \
 --test_mode dev_eval \
 --checkpoint_epoch 20 \
 --data_mode treebanks \
 --load_mode reuse \
 2>&1 | tee -a supertagging_test_$DATA.log
```

## Use the Supertagger
```
cd ccg_supertagger
bash run_supertagger.sh
```

**NOTE 1**: Please remember to update `--treebanks_dev_data_paths` and `--treebanks_test_data_paths` when updating the data of treebanks.<br>
**NOTE 2**: Please specify different parameters in `run_supertagger.sh` so as to use different functions.

### Important Parameters
`--lexical_category2idx_path`: The relative path to the dictionary mapping each CCG category to its index. Please specify either `../data/lexical_category2idx_cutoff_ccgbank.json` or `../data/lexical_category2idx_cutoff_treebanks.json`.<br>
`--model_dir`: The relative path to the directory of the pretrained language model. Default to `../plms/mt5-base`.<br>
`--checkpoint_path`: The path to the designated checkpoint file.<br>
`--device`: The device to run the experiment, default to be `cuda:0`.<br>
`--batch_size`: Default to be `8`.<br>
`--top_k`: The maximum number of supertags allowed for one word, default to `10`.<br>
`--beta`: The coefficient used to prune predicted categories, default to `0.0005`.<br>
`--mode`: The mode of the supertagger, choices include `predict` and `sanity_check`. If `predict`, you can specify the .json file path where you put a list of pretokenized sentences using `--pretokenized_sents_path` (default to `../data/pretokenized_sents.json`), and you should also specify the output file path with `--batch_predicted_path` (default to `./batch_predicted_supertags.json`). If `sanity_check`, the supertagger will just run on the designated data and print the (multi)tagging accuracy and average number of categories per word. Default to `sanity_check`.<br>
`--sanity_check_mode`: Used when `--mode` is specified `sanity_check`. Choices include `single_treebank`, `all_treebanks_dev`, `all_treebanks_test`, `ccgbank_dev` and `ccgbank_test`. The supertagger will first load the designated data from corresponding .auto files. When specifying `single_treebank`, please modify `--sanity_check_data_path` to choose the path to the tested treebank (default to `../treebanks/UD_English-Atis/en_atis-ud-dev.auto`).

### Some Example Scripts
 - Using the checkpoint file `./checkpoints_ccgbank/fc_mt5-base_drop0.5_epoch_8.pt` in the mode `sanity_check`, testing dev data of CCGBank.
```
#!/bin/bash

DATA="ccgbank"
TOPK=10
BETA=0.0005

python -u supertagger.py \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_ccgbank.json \
 --model_dir ../plms/mt5-base \
 --checkpoint_path ./checkpoints_ccgbank/fc_mt5-base_drop0.5_epoch_8.pt \
 --device cuda:0 \
 --batch_size 8 \
 --top_k ${TOPK} \
 --beta ${BETA} \
 --mode sanity_check \
 --sanity_check_mode ccgbank_dev \
 2>&1 | tee -a supertagger_${DATA}_${TOPK}_${BETA}.log
```

 - Using the checkpoint file `./checkpoints_treebanks/fc_mt5-base_drop0.5_epoch_20.pt` in the mode `sanity_check`, testing dev data of `UD_Chinese-GSD`.
```
#!/bin/bash

DATA="single_treebank"
TOPK=10
BETA=0.0005

python -u supertagger.py \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_treebanks.json \
 --model_dir ../plms/mt5-base \
 --checkpoint_path ./checkpoints_treebanks/fc_mt5-base_drop0.5_epoch_20.pt \
 --device cuda:0 \
 --batch_size 8 \
 --top_k ${TOPK} \
 --beta ${BETA} \
 --mode sanity_check \
 --sanity_check_mode single_treebank \
 --sanity_check_data_path ../treebanks/UD_Chinese-GSD/zh_gsd-ud-dev.auto \
 2>&1 | tee -a supertagger_${DATA}_${TOPK}_${BETA}.log
```

 - Using the checkpoint file `./checkpoints_treebanks/fc_mt5-base_drop0.5_epoch_20.pt` in the mode `sanity_check`, testing dev data of all treebanks.
```
#!/bin/bash

DATA="treebanks"
TOPK=10
BETA=0.0005

python -u supertagger.py \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_treebanks.json \
 --model_dir ../plms/mt5-base \
 --checkpoint_path ./checkpoints_treebanks/fc_mt5-base_drop0.5_epoch_20.pt \
 --device cuda:0 \
 --batch_size 8 \
 --top_k ${TOPK} \
 --beta ${BETA} \
 --mode sanity_check \
 --sanity_check_mode all_treebanks_dev \
 2>&1 | tee -a supertagger_${DATA}_${TOPK}_${BETA}.log
```

 - After using the checkpoint file `./checkpoints_treebanks/fc_mt5-base_drop0.5_epoch_20.pt` in the mode `predict`, the supertagger will load the pretokenized sentences from `--pretokenized_sents_path`, return all possible supertags for each token, and save them to `--batch_predicted_path`.
```
#!/bin/bash

MODE="predict"
TOPK=10
BETA=0.0005

python -u supertagger.py \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_treebanks.json \
 --model_dir ../plms/mt5-base \
 --checkpoint_path ./checkpoints_treebanks/fc_mt5-base_drop0.5_epoch_20.pt \
 --device cuda:0 \
 --batch_size 8 \
 --top_k ${TOPK} \
 --beta ${BETA} \
 --mode predict \
 2>&1 | tee -a supertagger_${MODE}_${TOPK}_${BETA}.log
```

# Parsing
This parser is built upon A* supertag-factored algorithms, i.e., it uses the scores from supertagging to conduct A* parsing.
## Use the parser
```
cd parsing
bash run_parser.sh
```

**NOTE 1**: Please remember to update `--treebanks_dev_data_paths` and `--treebanks_test_data_paths` when updating the data of treebanks.<br>
**NOTE 2**: Please specify different parameters in `run_parser.sh` so as to use different functions.

### Important Parameters
`--lexical_category2idx_path`: The relative path to the dictionary mapping each CCG category to its index. Please specify either `../data/lexical_category2idx_cutoff_ccgbank.json` or `../data/lexical_category2idx_cutoff_treebanks.json`. Default to `../data/lexical_category2idx_cutoff_treebanks.json`.<br>
`--instantiated_unary_rules_path`: The relative path to the .json file storing instantiated unary rules. Default to `../data/instantiated_unary_rules.json`.<br>
`--unary_rules_n`: The number of unary rules to use (the first n rules sorted according to frequency in the corpora), default to `20`.<br>
`--instantiated_binary_rules_path`: The relative path to the .json file storing instantiated binary rules, default to `../data/instantiated_binary_rules.json`.
 - If you need to build new instantiated unary rules and binary rules from treebanks. Please first go to `./data/utils.py`, and follow the script in the main function to generate `instantiated_unary_rules_raw.json` and `instantiated_binary_rules_raw.json`. Then go to `multiCCG/tools.py`, and run the `collect_unary_rules` and `collect_binary_rules` functions following the sample use in the main function. The resulting unary rules are sorted according to their frequency in the corpora.

`--supertagging_model_dir`: The relative path to the directory stoing the PLM of the supertagging model. Default to `../plms/mt5-base`.<br>
`--supertagging_model_checkpoint_path`: The relative path to the designated supertagging checkpoint file.<br>
`--predicted_auto_files_dir`: The relative path the directory storing predicted .auto files. Default to `./evaluation`.<br>
`--apply_supertagging_pruning` / `--no-apply_supertagging_pruning`: To control whether to apply the supertagging pruning method, if True, please specify `--beta`, default to `--apply_supertagging_pruning`.<br>
`--top_k_supertags`: The maximum number of supertags allowed for one word, default to `10`.<br>
`--beta`: The coefficient to prune predicted categories whose probabilities lie within $\beta$ of that of the best category, default to `0.0005`.<br>
`--batch_size`: The number of sentences to parse in a batch. Default to be `10`.<br>
`--possible_roots`: Allowed categories at the root of one parse, default to `S|NP|S/NP|S\\NP` for treebanks. Please specify `"S[dcl]|NP|S[wq]|S[q]|S[qem]|S[b]\\NP"` for CCGBank.<br>
`--device`: The device to conduct parsing, default to `cuda:0`.<br>
`--mode`: The mode of the parser, choices include `sanity_check`, `predict_sent`, `batch_sanity_check` and `predict_batch`. If `sanity_check`, the parser reads data from `--sample_data_path` (default to `../data/sample.auto`) and returns the parsing result with its golden supertags. If `predict_sent`, the parser reads data from `--sample_data_path` and returns the parsing result after supertagging. If `batch_sanity_check` or `predict_batch`, the parser reads in data designated by `batch_data_mode`, saves the predicted .auto files using their golden or predicted supertags under `./parsing/evaluation`, and the parseval scores under `./parsing/evaluation` folder. No default value for `--mode` and it must be specified.<br>
`--batch_data_mode`: Used when `--mode` is specified `batch_sanity_check` or `predict_batch`. Choices include `treebanks_dev`, `treebanks_test`, `ccgbank_dev` and `ccgbank_test`. No default value and it must be specified when used.

### Some Example Scripts
 - Sanity checking `../data/sample.auto`, with the supertagging model trained from **CCGBank**.
 ```
#!/bin/bash

DATA="ccgbank"
MODEL_NAME="mt5base"
TOPK=10
BETA=0.0005

python -u parser.py \
 --sample_data_path ../data/sample.auto \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_ccgbank.json \
 --instantiated_unary_rules_path ../data/instantiated_unary_rules_ccgbank.json \
 --instantiated_binary_rules_path ../data/instantiated_binary_rules_ccgbank.json \
 --unary_rules_n 20 \
 --supertagging_model_dir ../plms/mt5-base \
 --supertagging_model_checkpoint_path ../ccg_supertagger/checkpoints_ccgbank/fc_mt5-base_drop0.5_epoch_8.pt \
 --predicted_auto_files_dir ./evaluation \
 --beta ${BETA} \
 --top_k_supertags ${TOPK} \
 --batch_size 10 \
 --possible_roots "S[dcl]|NP|S[wq]|S[q]|S[qem]|S[b]\\NP" \
 --mode sanity_check \
 2>&1 | tee -a AStarParsing_${DATA}_${MODEL_NAME}_${TOPK}_${BETA}.log
 ```

 - Predicting parses for `../data/sample.auto`, with the supertagging model trained from **treebanks**.
 ```
#!/bin/bash

DATA="treebanks"
MODEL_NAME="mt5base"
TOPK=10
BETA=0.0005

python -u parser.py \
 --sample_data_path ../data/sample.auto \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_treebanks.json \
 --instantiated_unary_rules_path ../data/instantiated_unary_rules_treebanks.json \
 --instantiated_binary_rules_path ../data/instantiated_binary_rules_treebanks.json \
 --unary_rules_n 20 \
 --supertagging_model_dir ../plms/mt5-base \
 --supertagging_model_checkpoint_path ../ccg_supertagger/checkpoints_treebanks/fc_mt5-base_drop0.5_epoch_20.pt \
 --predicted_auto_files_dir ./evaluation \
 --beta ${BETA} \
 --top_k_supertags ${TOPK} \
 --batch_size 10 \
 --possible_roots "S|NP|S/NP|S\\NP" \
 --mode predict_sent \
 2>&1 | tee -a AStarParsing_${DATA}_${MODEL_NAME}_${TOPK}_${BETA}.log
 ```

 - Batch sanity checking dev data of **CCGBank**.
```
#!/bin/bash

DATA="ccgbank"
MODEL_NAME="mt5base"
TOPK=10
BETA=0.0005

python -u parser.py \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_ccgbank.json \
 --instantiated_unary_rules_path ../data/instantiated_unary_rules_ccgbank.json \
 --instantiated_binary_rules_path ../data/instantiated_binary_rules_ccgbank.json \
 --unary_rules_n 20 \
 --supertagging_model_dir ../plms/mt5-base \
 --supertagging_model_checkpoint_path ../ccg_supertagger/checkpoints_ccgbank/fc_mt5-base_drop0.5_epoch_8.pt \
 --predicted_auto_files_dir ./evaluation \
 --beta ${BETA} \
 --top_k_supertags ${TOPK} \
 --batch_size 10 \
 --possible_roots "S[dcl]|NP|S[wq]|S[q]|S[qem]|S[b]\\NP" \
 --mode batch_sanity_check \
 --batch_data_mode ccgbank_dev \
 2>&1 | tee -a AStarParsing_${DATA}_${MODEL_NAME}_${TOPK}_${BETA}.log
 ```

 - Batch predicting parses for dev data of **treebanks**.
```
#!/bin/bash

DATA="treebanks"
MODEL_NAME="mt5base"
TOPK=10
BETA=0.0005

python -u parser.py \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_treebanks.json \
 --instantiated_unary_rules_path ../data/instantiated_unary_rules_treebanks.json \
 --instantiated_binary_rules_path ../data/instantiated_binary_rules_treebanks.json \
 --unary_rules_n 20 \
 --supertagging_model_dir ../plms/mt5-base \
 --supertagging_model_checkpoint_path ../ccg_supertagger/checkpoints_treebanks/fc_mt5-base_drop0.5_epoch_20.pt \
 --predicted_auto_files_dir ./evaluation \
 --beta ${BETA} \
 --top_k_supertags ${TOPK} \
 --batch_size 10 \
 --possible_roots "S|NP|S/NP|S\\NP" \
 --mode predict_batch \
 --batch_data_mode treebanks_dev \
 2>&1 | tee -a AStarParsing_${DATA}_${MODEL_NAME}_${TOPK}_${BETA}.log
```