// Data constants for the application

// Breed data from the CSV files
const BREED_DATA = {
    1: { // Dogs
        1: "Affenpinscher",
        2: "Afghan Hound", 
        3: "Airedale Terrier",
        4: "Akbash",
        5: "Akita",
        6: "Alaskan Malamute",
        7: "American Bulldog",
        8: "American Eskimo Dog",
        9: "American Hairless Terrier",
        10: "American Staffordshire Terrier",
        11: "American Water Spaniel",
        12: "Anatolian Shepherd",
        13: "Appenzell Mountain Dog",
        14: "Australian Cattle Dog/Blue Heeler",
        15: "Australian Kelpie",
        16: "Australian Shepherd",
        17: "Australian Terrier",
        18: "Basenji",
        19: "Basset Hound",
        20: "Beagle",
        21: "Bearded Collie",
        22: "Beauceron",
        23: "Bedlington Terrier",
        24: "Belgian Shepherd Dog Sheepdog",
        25: "Belgian Shepherd Laekenois",
        26: "Belgian Shepherd Malinois",
        27: "Belgian Shepherd Tervuren",
        28: "Bernese Mountain Dog",
        29: "Bichon Frise",
        30: "Black and Tan Coonhound",
        31: "Black Labrador Retriever",
        32: "Black Mouth Cur",
        33: "Black Russian Terrier",
        34: "Bloodhound",
        35: "Blue Lacy",
        36: "Bluetick Coonhound",
        37: "Boerboel",
        38: "Bolognese",
        39: "Border Collie",
        40: "Border Terrier",
        41: "Borzoi",
        42: "Boston Terrier",
        43: "Bouvier des Flanders",
        44: "Boxer",
        45: "Boykin Spaniel",
        46: "Briard",
        47: "Brittany Spaniel",
        48: "Brussels Griffon",
        49: "Bull Terrier",
        50: "Bullmastiff",
        51: "Cairn Terrier",
        52: "Canaan Dog",
        53: "Cane Corso Mastiff",
        54: "Carolina Dog",
        55: "Catahoula Leopard Dog",
        56: "Cattle Dog",
        57: "Caucasian Sheepdog (Caucasian Ovtcharka)",
        58: "Cavalier King Charles Spaniel",
        59: "Chesapeake Bay Retriever",
        60: "Chihuahua",
        61: "Chinese Crested Dog",
        62: "Chinese Foo Dog",
        63: "Chinook",
        64: "Chocolate Labrador Retriever",
        65: "Chow Chow",
        66: "Cirneco dell'Etna",
        67: "Clumber Spaniel",
        68: "Cockapoo",
        69: "Cocker Spaniel",
        70: "Collie",
        71: "Coonhound",
        72: "Corgi",
        73: "Coton de Tulear",
        74: "Curly-Coated Retriever",
        75: "Dachshund",
        76: "Dalmatian",
        77: "Dandi Dinmont Terrier",
        78: "Doberman Pinscher",
        79: "Dogo Argentino",
        80: "Dogue de Bordeaux",
        81: "Dutch Shepherd",
        82: "English Bulldog",
        83: "English Cocker Spaniel",
        84: "English Coonhound",
        85: "English Pointer",
        86: "English Setter",
        87: "English Shepherd",
        88: "English Springer Spaniel",
        89: "English Toy Spaniel",
        90: "Entlebucher",
        91: "Eskimo Dog",
        92: "Feist",
        93: "Field Spaniel",
        94: "Fila Brasileiro",
        95: "Finnish Lapphund",
        96: "Finnish Spitz",
        97: "Flat-coated Retriever",
        98: "Fox Terrier",
        99: "Foxhound",
        100: "French Bulldog",
        101: "Galgo Spanish Greyhound",
        102: "German Pinscher",
        103: "German Shepherd Dog",
        104: "German Shorthaired Pointer",
        105: "German Spitz",
        106: "German Wirehaired Pointer",
        107: "Giant Schnauzer",
        108: "Glen of Imaal Terrier",
        109: "Golden Retriever",
        110: "Gordon Setter",
        111: "Great Dane",
        112: "Great Pyrenees",
        113: "Greater Swiss Mountain Dog",
        114: "Greyhound",
        115: "Harrier",
        116: "Havanese",
        117: "Hound",
        118: "Hovawart",
        119: "Husky",
        120: "Ibizan Hound",
        121: "Illyrian Sheepdog",
        122: "Irish Setter",
        123: "Irish Terrier",
        124: "Irish Water Spaniel",
        125: "Irish Wolfhound",
        126: "Italian Greyhound",
        127: "Italian Spinone",
        128: "Jack Russell Terrier",
        129: "Jack Russell Terrier (Parson Russell Terrier)",
        130: "Japanese Chin",
        131: "Jindo",
        132: "Kai Dog",
        133: "Karelian Bear Dog",
        134: "Keeshond",
        135: "Kerry Blue Terrier",
        136: "Kishu",
        137: "Klee Kai",
        138: "Komondor",
        139: "Kuvasz",
        140: "Kyi Leo",
        141: "Labrador Retriever",
        142: "Lakeland Terrier",
        143: "Lancashire Heeler",
        144: "Leonberger",
        145: "Lhasa Apso",
        146: "Lowchen",
        147: "Maltese",
        148: "Manchester Terrier",
        149: "Maremma Sheepdog",
        150: "Mastiff",
        151: "McNab",
        152: "Miniature Pinscher",
        153: "Mountain Cur",
        154: "Mountain Dog",
        155: "Munsterlander",
        156: "Neapolitan Mastiff",
        157: "New Guinea Singing Dog",
        158: "Newfoundland Dog",
        159: "Norfolk Terrier",
        160: "Norwegian Buhund",
        161: "Norwegian Elkhound",
        162: "Norwegian Lundehund",
        163: "Norwich Terrier",
        164: "Nova Scotia Duck-tolling Retriever",
        165: "Old English Sheepdog",
        166: "Otterhound",
        167: "Papillon",
        168: "Pekingese",
        169: "Pharaoh Hound",
        170: "Pit Bull Terrier",
        171: "Plott Hound",
        172: "Pointer",
        173: "Pomeranian",
        174: "Poodle",
        175: "Portuguese Water Dog",
        176: "Pug",
        177: "Puli",
        178: "Pumi",
        179: "Pyrenean Shepherd",
        180: "Rat Terrier",
        181: "Redbone Coonhound",
        182: "Rhodesian Ridgeback",
        183: "Rottweiler",
        184: "Russian Toy",
        185: "Saint Bernard",
        186: "Saluki",
        187: "Samoyed",
        188: "Schipperke",
        189: "Scottish Deerhound",
        190: "Scottish Terrier",
        191: "Sealyham Terrier",
        192: "Shar Pei",
        193: "Shetland Sheepdog",
        194: "Shiba Inu",
        195: "Shih Tzu",
        196: "Siberian Husky",
        197: "Silky Terrier",
        198: "Skye Terrier",
        199: "Sloughi",
        200: "Smooth Fox Terrier",
        201: "Soft Coated Wheaten Terrier",
        202: "Spinone Italiano",
        203: "Staffordshire Bull Terrier",
        204: "Standard Schnauzer",
        205: "Sussex Spaniel",
        206: "Swedish Vallhund",
        207: "Tibetan Mastiff",
        208: "Tibetan Spaniel",
        209: "Tibetan Terrier",
        210: "Toy Fox Terrier",
        211: "Treeing Walker Coonhound",
        212: "Vizsla",
        213: "Weimaraner",
        214: "Welsh Corgi",
        215: "Welsh Springer Spaniel",
        216: "Welsh Terrier",
        217: "West Highland White Terrier",
        218: "Whippet",
        219: "Wire Fox Terrier",
        220: "Wirehaired Pointing Griffon",
        221: "Xoloitzcuintli",
        222: "Yorkshire Terrier",
        307: "Mixed Breed" 
    },
    2: { // Cats
        223: "Abyssinian",
        224: "American Bobtail",
        225: "American Curl",
        226: "American Shorthair",
        227: "American Wirehair",
        228: "Balinese",
        229: "Bengal",
        230: "Birman",
        231: "Bombay",
        232: "British Longhair",
        233: "British Shorthair",
        234: "Burmese",
        235: "Burmilla",
        236: "California Spangled",
        237: "Chartreux",
        238: "Chausie",
        239: "Cornish Rex",
        240: "Cymric",
        241: "Devon Rex",
        242: "Domestic Longhair",
        243: "Domestic Medium Hair",
        244: "Domestic Shorthair",
        245: "Egyptian Mau",
        246: "Exotic Shorthair",
        247: "Havana",
        248: "Himalayan",
        249: "Japanese Bobtail",
        250: "Javanese",
        251: "Korat",
        252: "LaPerm",
        253: "Maine Coon",
        254: "Manx",
        255: "Munchkin",
        256: "Nebelung",
        257: "Norwegian Forest Cat",
        258: "Ocicat",
        259: "Oriental Longhair",
        260: "Oriental Shorthair",
        261: "Persian",
        262: "Pixie-bob",
        263: "Ragamuffin",
        264: "Ragdoll",
        265: "Russian Blue",
        266: "Scottish Fold",
        267: "Selkirk Rex",
        268: "Siamese",
        269: "Siberian",
        270: "Singapura",
        271: "Snowshoe",
        272: "Somali",
        273: "Sphynx",
        274: "Tonkinese",
        275: "Turkish Angora",
        276: "Turkish Van"
    }
};

// Color data
const COLOR_DATA = {
    1: "Black",
    2: "Brown",
    3: "Golden",
    4: "Yellow",
    5: "Cream",
    6: "Gray",
    7: "White",
    8: "Red",
    9: "Blue",
    10: "Orange",
    11: "Chocolate",
    12: "Silver",
    13: "Tan",
    14: "Brindle",
    15: "Merle",
    16: "Calico",
    17: "Tricolor",
    18: "Spotted",
    19: "Tabby",
    20: "Tortoiseshell"
};

// Malaysian states data
const STATE_DATA = {
    41336: "Johor",
    41325: "Kedah", 
    41367: "Kelantan",
    41401: "Kuala Lumpur",
    41415: "Labuan",
    41324: "Melaka",
    41332: "Negeri Sembilan",
    41335: "Pahang",
    41330: "Perak",
    41380: "Perlis",
    41327: "Pulau Pinang",
    41345: "Sabah",
    41342: "Sarawak", 
    41326: "Selangor",
    41361: "Terengganu"
};

// Adoption time periods for prediction
const ADOPTION_PERIODS = {
    0: {
        label: "Same Day - 1 Week",
        description: "This pet is predicted to be adopted within the first week of listing.",
        days: "0-7 days",
        badge: "fast"
    },
    1: {
        label: "1 Week - 1 Month", 
        description: "This pet is predicted to be adopted within 8-30 days of listing.",
        days: "8-30 days",
        badge: "fast"
    },
    2: {
        label: "1-3 Months",
        description: "This pet is predicted to be adopted within 31-90 days of listing.",
        days: "31-90 days", 
        badge: "moderate"
    },
    3: {
        label: "3+ Months",
        description: "This pet may take more than 100 days to find their forever home.",
        days: "100+ days",
        badge: "slow"
    }
};

// Export data for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        BREED_DATA,
        COLOR_DATA,
        STATE_DATA,
        ADOPTION_PERIODS
    };
}
