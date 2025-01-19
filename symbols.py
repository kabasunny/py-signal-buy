import random

symbols = [
    "1570",  "7201", "6146", "7013", "7203", "5803", "2432", "6857", "7012", "6920", "7011", "7267", "9984", "8035", "8306", "9983", "2914", "4204", "4755", "7003", 
    # "6526", "8136", "8058", "8316", "6758", "8411", "1357", "7974", "7211", "4107", "9101", "6861", "4063", "9432", "3778", "6501", "1321", "6594", "6098", "3697", 
    # "3197", "1360", "6254", "8766", "9104", "4502", "5253", "1605", "3498", "6902", "2702", "9501", "3099", "8001", "1458", "5108", "6723", "5801", "297A", "7751", 
    # "6367", "285A", "1579", "9843", "8031", "5401", "7735", "9503", "4661", "4568", "302A", "9107", "9434", "6981", "9433", "3382", "219A", "2160", "6762", "6273", 
    # "4519", "215A", "6503", "7936", "8725", "1459", "4385", "6301", "8053", "7261", "6954", "7270", "6201", "7532", "4523", "7272", "4578", "6702", "3003", "8002", 
    # "7741", "6752", "1365", "3086", "3696", "6701", "5411", "4503", "7182", "5105", "7453", "5802", "8604", "1911", "3350", "8750", "6315", "4543", "1928", "8802", 
    # "9024", "7269", "9020", "9201", "8473", "5201", "6971", "8308", "8591", "3549", "6963", "6525", "4911", "1655", "7832", "7205", "6988", "2502", "2503", "4901", 
    # "9613", "9022", "1568", "5406", "7014", "4452", "9468", "3624", "8015", "6645", "7746", "8267", "2413", "5020", "3097", "5713", "5631", "8309", "9766", "6976", 
    # "4062", "1925", "8985", "8801", "7733", "9023", "9166", "6178", "2768", "3436", "6460", "9202", "6326", "1306", "8804", "8830", "8593", "6141", "2802", "6323", 
    # "7259", "4704", "4477", "8795", "4188", "9416", "1571", "8630", "3405", "6506", "6871", "4784", "1663", "7342", "8233", "8601", "6324", "7220", "2558", "9509", 
    # "4967", "3064", "6967", "4324", "8698", "5344", "8783", "4004", "2244", "7867", "7202", "5032", "9735", "9021", "7163", "4307", "4631", "4521", "4666", "2501", 
    # "2267", "8179", "4776", "6361", "5101", "7912", "4527", "4689", "6728", "1812", "9506", "8894", "3133", "4676", "1489", "6504", "8113", "7309", "2621", "304A", 
    # "4935", "6532", "6993", "4576", "3659", "4507", "1320", "9672", "9508", "2269", "4183", "9531", "6740", "4528", "4680", "4922", "9502", "3092", "6481", "3407", 
    # "2875", "9348", "1330", "6590", "5838", "1801", "4583", "7186", "4927", "6103", "3038", "190A", "5535", "7911", "9697", "1367", "1802", "1540", "278A", "7518", 
    # "5334", "4902", "2897", "2801", "3402", "9602", "3088", "1803", "1329", "5110", "7956", "3563", "8697", "6965", "6383", "9719", "4912", "9532", "6305", "1366", 
    # "6724", "4005", "3993", "5805", "2247", "2986", "1346", "3105", "5711", "1308", "9504", "9147", "3861", "3994", "4293", "5726", "1547", "7550", "7752", "2840", 
    # "1545", "4587", "4684", "5019", "3626", "9009", "4751", "9005", "260A", "5938", "4480", "6586", "5301", "3349", "3231", "8331", "4768", "5332", "3289", "9706", 
    # "6754", "298A", "3185", "3028", "6368", "9684", "8011", "6736", "6869", "8304", "8963", "3793", "4980", "7203", "6758", "9433", "5803"
]

def get_train_and_real_data_symbols(train_ratio=0.7):
    random.shuffle(symbols)
    train_size = int(train_ratio * len(symbols))
    train_symbols = symbols[:train_size]
    real_data_symbols = symbols[train_size:]
    return train_symbols, real_data_symbols