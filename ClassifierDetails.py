#Credit: https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
stop_words = [
"a", "about", "above", "across", "after", "afterwards",
"again", "all", "almost", "alone", "along", "already", "also",
"although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
"this", "those", "though", "through", "throughout",
"thru", "thus", "to", "together", "too", "toward", "towards",
"under", "until", "up", "upon", "us",
"very", "was", "we", "well", "were", "what", "whatever", "when",
"whence", "whenever", "where", "whereafter", "whereas", "whereby",
"wherein", "whereupon", "wherever", "whether", "which", "while",
"who", "whoever", "whom", "whose", "why", "will", "with",
"within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
]

usernames_list = ["apurdygoldendoodle",
                  "bambergbob",
                  "olivia_ann_morris",
                  "vickymaythebirdofprey",
                  "hannahlhawkins",
                  "zsuzsanna_petho",
                  "andrewbylina",
                  "elisafrompisa",
                  "olivia.hxlmes",
                  "benfish_",
                  "moi_taiga",
                  "tobypastures",
                  "__mell.y.__",
                  "gennitran",
                  "kiwibear_",
                  "jonjohnieee",
                  "anwaythisside",
                  "clemence.rvlln",
                  "__callmedana",
                  "dr.hoomir",
                  "lopezzemelie",
                  "crazee_canadia",
                  "adamrollo",
                  "_kasia9004_",
                  "mariaannagialdi",
                  "tsamneb",
                  "katemariexo1",
                  "odh_xx",
                  "alisawinkler",
                  "forgoodnesskate",
                  "definitelyfrench",
                  "dionedanielle",
                  "lallaboooo",
                  "_jovvita_",
                  "carolina_uk_",
                  "mary_underheaven",
                  "_mafact_",
                  "kristy.ehmer",
                  "franco.corleonee",
                  "apriljjackson",
                  "irisdy01",
                  "rungymyoga",
                  "lewis_jackson97",
                  "sasseekassblog",
                  "_.denise.marie._",
                  "darrsea",
                    "laurennoneill",
                  "the_aesthete_girl",
                  "jonathanmillerjam",
                  "maddie.in.motion",
                  "hanuhkate",
                  "lucia.v.83",
                  "ecenayman",
                  "mayal_ondon",
                  "paynejack_",
                  "joshualindeberg",
                  "eela.ne",
                  "benrussell47photography",
                  "mers8655",
                  "giancarlomir",
                  "melhrn_",
                  "cakesncycles",
                    "rebecca.lxx",
                    "kirkilou7",
                    "shulinsciene",
                    "abi_dickmann",
                    "vickywintgen",
                    "kamalrazvi",
                  "ra.allexandra",
                  "datcassdoeee",
                  "dallafit_doza",
                  "meredithtx",
                  "akbss553534",
                  "manuelfrigerio.real",
                  "deniskozhukhovofficial",
                  "sgtguerrero.usmc",
                  "satuzainen",
                  "danamly",
                  "nadine_r_1207",
                  "zackm_96",
                  "manontravelaroundx",
                  "chris_momo_lifestyle",
                  "nataliereneekirk",
                  "jaimiegardner",
                  "hannah_woodcock",
                  "itsbeyzo",
                  "anitarunner",
                  "shieeef",
                  "laurentessa12",
                  "zombimo",
                  "hleejla",
                  "skylerstorywwe",
                  "danijelaknezevic",
                  "adutchrobin",
                  "aniamosiejczuk",
                  "anushree.bhattacharya_",
                  "lozzy_locks",
                  "geriee_berry",
                  "czar_domka",
                  "kelseyredmore",
                  "miriam_chiesa",
                  "_martini_93",
                  "livwalde",
                  "slmlmg",
                  "pernillaottosson",
                  "beccaejohns",
                  "annafinessi",
                  "julesrees19",
                  "raul_k07",
                  "marisemeniuk",
                  "ldnbyshane",
                  "skoneczs",
                  "kirikaija",
                  "czarnula19coco",
                  "sorayaeckes",
                  "melaniehfl",
                  "kuzniarska_acha",
                  "thenorthernist",
                  "lucia.v.83",
                  "iphy17",
                  "ana_bistis_g",
                  "24mao",
                  "wheres.lea",
                  "lola.roses",
                  "yazminesamthevegan",
                  "londonviewpoints",
                  "intibint",
                  "laurengracelife",
                  "amandabaldin",
                  "alinabarsx",
                  "paaticzka",
                  "yuka.ohnox",
                  "jony_yagua",
                  "rhian_nicola7",
                  "theesarahpowell",
                  "just_anemarie_photography",
                  "liss.pups",
                  "abi_lou89",
                  "doctor.gerardo",
                  "evarbcast",
                  "cristinacristea27",
                  "nnfsnapshots",
                  "sammigrundyx",
                  "angelo.carlucci",
                  "yuka.ohnox",
                  "rhian_nicola7",
                  "sadiemaltby",
                  "vikki.holt.xx",
                  "paaticzka",
                  "evarbcast",
                  "penninm7",
                  "fran.ncesca",
                  "mo.n.i.e",
                  "missjette",
                  "littlebaby.dreamer",
                  "miss_ch3rry_myers",
                  "dimitra_zisimou",
                  "lillafamiljen_",
                  "marika_photomodel",
                  "justinliftsthings",
                  "steffflora",

                  ]
