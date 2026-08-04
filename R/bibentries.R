bibentries = c(# nolint start
  gorishniy2021revisiting = bibentry("article",
    title = "Revisiting Deep Learning for Tabular Data",
    author = c(
      person("Yury", "Gorishniy"),
      person("Ivan", "Rubachev"),
      person("Valentin", "Khrulkov"),
      person("Artem", "Babenko")
    ),
    journal = "arXiv",
    volume = "2106.11959",
    year = "2021",
  ),
  devlin2018bert = bibentry("article",
    title = "Bert: Pre-training of deep bidirectional transformers for language understanding",
    author = c(
      person("Jacob", "Devlin"),
      person("Ming-Wei", "Chang"),
      person("Kenton", "Lee"),
      person("Kristina", "Toutanova")
    ),
    journal = "arXiv preprint arXiv:1810.04805",
    year = "2018"
  ),
  vaswani2017attention = bibentry("article",
    title = "Attention is all you need",
    author = c(
      person("Ashish", "Vaswani"),
      person("Noam", "Shazeer"),
      person("Niki", "Parmar"),
      person("Jakob", "Uszkoreit"),
      person("Llion", "Jones"),
      person("Aidan N", "Gomez"),
      person("\xc5\x81ukasz", "Kaiser"),
      person("Illia", "Polosukhin")
    ),
    journal = "Advances in neural information processing systems",
    volume = "30",
    year = "2017"
  ),
  arik2021tabnet = bibentry("inproceedings",
    title = "Tabnet: Attentive interpretable tabular learning",
    author = c(
      person("Sercan O", "Ar\xc4\xb1k"),
      person("Tomas", "Pfister")
    ),
    booktitle = "AAAI",
    volume = "35",
    pages = "6679--6687",
    year = "2021"
  ),

  ioffe2015batch = bibentry("inproceedings",
    title = "Batch normalization: Accelerating deep network training by reducing internal covariate shift",
    author = c(
      person("Sergey", "Ioffe"),
      person("Christian", "Szegedy")
    ),
    booktitle = "International conference on machine learning",
    pages = "448--456",
    year = "2015",
    organization = "PMLR"
  ),
  krizhevsky2017imagenet = bibentry(
    "article",
    title = "Imagenet classification with deep convolutional neural networks",
    author = c(
      person("Alex", "Krizhevsky"),
      person("Ilya", "Sutskever"),
      person("Geoffrey E.", "Hinton")
    ),
    journal = "Communications of the ACM",
    volume = "60",
    number = "6",
    pages = "84--90",
    year = "2017",
    publisher = "Association for Computing Machinery (ACM)"
  ),
  imagenet2009 = bibentry(
    bibtype = "InProceedings",
    textVersion = "Deng, J. et al. (2009). Imagenet: A large-scale hierarchical image database. In Proceedings of the 2009 IEEE conference on computer vision and pattern recognition (pp. 248--255). IEEE.",
    header = "deng2009imagenet",
    title = "Imagenet: A large-scale hierarchical image database",
    author = c(
      person("Jia", "Deng"),
      person("Wei", "Dong"),
      person("Richard", "Socher"),
      person("Li-Jia", "Li"),
      person("Kai", "Li"),
      person("Li", "Fei-Fei")
    ),
    booktitle = "2009 IEEE conference on computer vision and pattern recognition",
    pages = "248--255",
    year = "2009",
    organization = "IEEE"
  ),
  mnist = bibentry("article",
    author = c(
      person("Y.", "Lecun"),
      person("L.", "Bottou"),
      person("Y.", "Bengio"),
      person("P.", "Haffner")
    ),
    journal = 'Proceedings of the IEEE',
    title = 'Gradient-based learning applied to document recognition',
    year = '1998',
    volume = '86',
    number = '11',
    pages = '2278-2324',
    doi = '10.1109/5.726791'
  ),
  anderson_1936 = bibentry("article",
    doi       = "10.2307/2394164",
    year      = "1936",
    month     = "9",
    publisher = "JSTOR",
    volume    = "23",
    number    = "3",
    pages     = "457",
    author    = person("Edgar", "Anderson"),
    title     = "The Species Problem in Iris",
    journal   = "Annals of the Missouri Botanical Garden"
  ),
  sandler2018mobilenetv2 = bibentry("InProceedings",
    title    = "Mobilenetv2: Inverted residuals and linear bottlenecks",
    author   = c(
      person("Mark", "Sandler"),
      person("Andrew", "Howard"),
      person("Menglong", "Zhu"),
      person("Andrey", "Zhmoginov"),
      person("Liang-Chieh", "Chen")
    ),
    booktitle= "Proceedings of the IEEE conference on computer vision and pattern recognition",
    pages    = "4510--4520",
    year     = "2018"
  ),
  simonyan2014very = bibentry("article",
    title  = "Very deep convolutional networks for large-scale image recognition",
    author = c(
      person("Karen", "Simonyan"),
      person("Andrew", "Zisserman")
    ),
    journal= "arXiv preprint arXiv:1409.1556",
    year   = "2014"
  ),
  he2016deep = bibentry("InProceedings",
    title     = "Deep residual learning for image recognition",
    author    = c(
      person("Kaiming", "He"),
      person("Xiangyu", "Zhang"),
      person("Shaoqing", "Ren"),
      person("Jian", "Sun")
    ),
    booktitle = "Proceedings of the IEEE conference on computer vision and pattern recognition",
    pages     = "770--778",
    year      = "2016"
  ),
  krizhevsky2014one = bibentry("article",
    title = "One weird trick for parallelizing convolutional neural networks",
    author = person("Alex", "Krizhevsky"),
    journal = "arXiv preprint arXiv:1404.5997",
    year = "2014"
  ),
  szegedy2016rethinking = bibentry("InProceedings",
    title     = "Rethinking the inception architecture for computer vision",
    author    = c(
      person("Christian", "Szegedy"),
      person("Vincent", "Vanhoucke"),
      person("Sergey", "Ioffe"),
      person("Jonathon", "Shlens"),
      person("Zbigniew", "Wojna")
    ),
    booktitle = "Proceedings of the IEEE conference on computer vision and pattern recognition",
    pages     = "2818--2826",
    year      = "2016"
  ),
  melanoma2021 = bibentry("article",
    title = "A patient-centric dataset of images and metadata for identifying melanomas using clinical context",
    author = c(
      person("V.", "Rotemberg"),
      person("N.", "Kurtansky"),
      person("B.", "Betz-Stablein"),
      person("L.", "Caffery"),
      person("E.", "Chousakos"),
      person("N.", "Codella"),
      person("M.", "Combalia"),
      person("S.", "Dusza"),
      person("P.", "Guitera"),
      person("D.", "Gutman"),
      person("A.", "Halpern"),
      person("B.", "Helba"),
      person("H.", "Kittler"),
      person("K.", "Kose"),
      person("S.", "Langer"),
      person("K.", "Lioprys"),
      person("J.", "Malvehy"),
      person("S.", "Musthaq"),
      person("J.", "Nanda"),
      person("O.", "Reiter"),
      person("G.", "Shih"),
      person("A.", "Stratigos"),
      person("P.", "Tschandl"),
      person("J.", "Weber"),
      person("P.", "Soyer")
    ),
    journal = "Scientific Data",
    volume = "8",
    pages = "34",
    year = "2021",
    doi = "10.1038/s41597-021-00815-z"
  ),
  cifar2009 = bibentry("article",
    title = "Learning Multiple Layers of Features from Tiny Images",
    author = person("Alex", "Krizhevsky"),
    journal= "Master's thesis, Department of Computer Science, University of Toronto",
    year = "2009"
  ),
  shazeer2020glu = bibentry("misc",
    title = "GLU Variants Improve Transformer",
    author = person("Noam", "Shazeer"),
    year = "2020",
    eprint = "2002.05202",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    url = "https://arxiv.org/abs/2002.05202"
  ),
  gorishniy2025tabm = bibentry("inproceedings",
    title = "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling",
    author = "Yury Gorishniy and Akim Kotelnikov and Artem Babenko",
    booktitle = "The Thirteenth International Conference on Learning Representations (ICLR)",
    year = "2025",
    eprint = "2410.24210",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    url = "https://openreview.net/forum?id=Sd4wYYOhmY"
  ),
  gorishniy2022embeddings = bibentry("inproceedings",
    title = "On Embeddings for Numerical Features in Tabular Deep Learning",
    author = "Yury Gorishniy and Ivan Rubachev and Artem Babenko",
    booktitle = "Advances in Neural Information Processing Systems 35 (NeurIPS)",
    year = "2022",
    eprint = "2203.05556",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    url = "https://arxiv.org/abs/2203.05556"
  ),
  wen2020batchensemble = bibentry("inproceedings",
    title = "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning",
    author = "Yeming Wen and Dustin Tran and Jimmy Ba",
    booktitle = "The Eighth International Conference on Learning Representations (ICLR)",
    year = "2020",
    eprint = "2002.06715",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    url = "https://openreview.net/forum?id=Sklf1yrYDr"
  ),
  xie2017aggregated = bibentry("InProceedings",
    title     = "Aggregated residual transformations for deep neural networks",
    author    = c(
      person("Saining", "Xie"),
      person("Ross", "Girshick"),
      person("Piotr", "Doll\xc3\xa1r"),
      person("Zhuowen", "Tu"),
      person("Kaiming", "He")
    ),
    booktitle = "Proceedings of the IEEE conference on computer vision and pattern recognition",
    pages     = "1492--1500",
    year      = "2017"
  ),
  zagoruyko2016wide = bibentry("InProceedings",
    title     = "Wide residual networks",
    author    = c(
      person("Sergey", "Zagoruyko"),
      person("Nikos", "Komodakis")
    ),
    booktitle = "Proceedings of the British Machine Vision Conference (BMVC)",
    year      = "2016",
    eprint    = "1605.07146",
    archivePrefix = "arXiv",
    primaryClass = "cs.CV",
    url       = "https://arxiv.org/abs/1605.07146"
  ),
  howard2019searching = bibentry("InProceedings",
    title     = "Searching for MobileNetV3",
    author    = c(
      person("Andrew", "Howard"),
      person("Mark", "Sandler"),
      person("Grace", "Chu"),
      person("Liang-Chieh", "Chen"),
      person("Bo", "Chen"),
      person("Mingxing", "Tan"),
      person("Weijun", "Wang"),
      person("Yukun", "Zhu"),
      person("Ruoming", "Pang"),
      person("Vijay", "Vasudevan"),
      person("Quoc V.", "Le"),
      person("Hartwig", "Adam")
    ),
    booktitle = "Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)",
    pages     = "1314--1324",
    year      = "2019"
  ),
  tan2019efficientnet = bibentry("InProceedings",
    title     = "EfficientNet: Rethinking model scaling for convolutional neural networks",
    author    = c(
      person("Mingxing", "Tan"),
      person("Quoc", "Le")
    ),
    booktitle = "Proceedings of the 36th International Conference on Machine Learning (ICML)",
    pages     = "6105--6114",
    year      = "2019",
    organization = "PMLR"
  ),
  tan2021efficientnetv2 = bibentry("InProceedings",
    title     = "EfficientNetV2: Smaller models and faster training",
    author    = c(
      person("Mingxing", "Tan"),
      person("Quoc", "Le")
    ),
    booktitle = "Proceedings of the 38th International Conference on Machine Learning (ICML)",
    pages     = "10096--10106",
    year      = "2021",
    organization = "PMLR"
  ),
  liu2022convnet = bibentry("InProceedings",
    title     = "A ConvNet for the 2020s",
    author    = c(
      person("Zhuang", "Liu"),
      person("Hanzi", "Mao"),
      person("Chao-Yuan", "Wu"),
      person("Christoph", "Feichtenhofer"),
      person("Trevor", "Darrell"),
      person("Saining", "Xie")
    ),
    booktitle = "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    pages     = "11976--11986",
    year      = "2022"
  ),
  dosovitskiy2021image = bibentry("InProceedings",
    title     = "An image is worth 16x16 words: Transformers for image recognition at scale",
    author    = c(
      person("Alexey", "Dosovitskiy"),
      person("Lucas", "Beyer"),
      person("Alexander", "Kolesnikov"),
      person("Dirk", "Weissenborn"),
      person("Xiaohua", "Zhai"),
      person("Thomas", "Unterthiner"),
      person("Mostafa", "Dehghani"),
      person("Matthias", "Minderer"),
      person("Georg", "Heigold"),
      person("Sylvain", "Gelly"),
      person("Jakob", "Uszkoreit"),
      person("Neil", "Houlsby")
    ),
    booktitle = "The Ninth International Conference on Learning Representations (ICLR)",
    year      = "2021",
    eprint    = "2010.11929",
    archivePrefix = "arXiv",
    primaryClass = "cs.CV",
    url       = "https://arxiv.org/abs/2010.11929"
  ),
  tu2022maxvit = bibentry("InProceedings",
    title     = "MaxViT: Multi-axis vision transformer",
    author    = c(
      person("Zhengzhong", "Tu"),
      person("Hossein", "Talebi"),
      person("Han", "Zhang"),
      person("Feng", "Yang"),
      person("Peyman", "Milanfar"),
      person("Alan C.", "Bovik"),
      person("Yinxiao", "Li")
    ),
    booktitle = "Proceedings of the European Conference on Computer Vision (ECCV)",
    pages     = "459--479",
    year      = "2022"
  )
) # nolint end
