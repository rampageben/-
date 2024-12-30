import torch
import torch.nn as nn
from collections import defaultdict
import pandas as pd
from datetime import time

class TradingStrategy:
    def __init__(self, model_path, length=5):
        """
        初始化交易策略物件
        :param model: 預測模型的物件
        """
        self.position = 0  # 目前持倉狀態 (0: 無倉, 1: 多頭, -1: 空頭)
        self.initialize_model(model_path)
        self.length = length
        self.counter = 0
        self.current_date = None
        self.state = False
        self.chose_number = 1
        self.trade = []
        self.hold = defaultdict(list)
        self.index = 0

    def initialize_model(self, model_path):
        """
        初始化模型參數
        :param model_params: 模型初始化所需的參數字典
        """
        self.model =  LSTM(input_size=5, output_size=4)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        self.model.eval()

    def data_load(self, time_data):
        """
        處理輸入的資料
        :param time_data: 輸入的多支股票的資料清單
        """
        # 提取日期（假設資料格式相同，使用第一筆資料的 'date_only'）
        incoming_date = time_data[0]['date_only']

        if self.current_date is None:
            self.current_date = incoming_date
            self.stock_data = defaultdict(list)
            self.stock_open = defaultdict(float)
            self.counter = 0
            self.state = False
        # 如果日期不同，清空變數
        elif self.current_date != incoming_date:
            print(f"日期變更，清空已存資料: {self.current_date} -> {incoming_date}")
            self.current_date = incoming_date
            self.stock_data = defaultdict(list)
            self.stock_open = defaultdict(float)
            self.counter = 0
            self.state = False
            gain,length = self.get_trade()
            print('gain',gain)
            print('length',length)

        # 確認股票數量是否正確
        if len(time_data) != self.length:
            raise ValueError(f"股票數量不正確！期望 {self.length} 支，實際收到 {len(time_data)} 支。 ")

        if self.state is False:
            # 儲存每支股票的資料
            for stock in time_data:
                tic = stock['tic']  # 股票代碼
                self.stock_data[tic].append(stock)
                if tic not in self.stock_open:
                    self.stock_open[tic] = stock.open
            self.counter += 1

            # print(f"資料已儲存，當前日期: {self.current_date}, {self.counter}")
            if self.counter == 30:
                self.data = self.data_processing()
                self.data = self.input_data_normalization(self.data)
                self.output = self.input_data_totensor(self.data)
                self.output = self.output_todata(self.output)
                self.org_output = self.get_org_data(self.output)
                self.index_data = self.get_indext(self.org_output)
                self.sorted_data = self.sort_with_indext(self.index_data)
                self.pay_data = self.get_pay_number(self.sorted_data)      
                self.state = True
               
        
        if self.state:
            # print('pay_data',self.pay_data)
            if len(self.pay_data) != 0:
                for stock in time_data:
                    tic = stock['tic']
                    if tic in self.pay_data and tic not in self.hold :
                        if len(self.trade) != 0 :
                            for n in range(1,self.chose_number+1):
                                if self.trade[self.index - n]['action'] == 'buying' and self.trade[self.index - n]['tic'] == tic:
                                    self.trade[self.index - n]['buy_date_detail'] = stock['date']
                                    self.trade[self.index - n]['buy_date'] = stock['date_only']
                                    self.hold[tic].append(self.trade[self.index - n])
                                    self.trade[self.index - n]['action'] = 'holding'
                                elif self.trade[self.index - n]['buy_date'] != stock['date_only'] and self.trade[self.index - n]['tic'] == tic:
                                    self.pay_chose(stock, self.pay_data)
                        else:
                            self.pay_chose(stock, self.pay_data)
                    elif tic in self.hold:
                        for n in range(1,self.chose_number+1):
                            if self.trade[self.index - n]['action'] == 'holding' and stock['date'].time() == time(13,30) and self.trade[self.index - n]['tic'] == tic:
                                self.trade[self.index - n]['sell_date_detail'] = stock['date']
                                self.trade[self.index - n]['sell_date'] = stock['date_only']
                                self.trade[self.index - n]['sell_price'] = stock['close']
                                
                                self.trade[self.index - n]['gain'] = round((self.trade[self.index - n]['sell_price'] - self.trade[self.index - n]['price']) * 1000,0 )
                                self.trade[self.index - n]['action'] = 'done'
                                self.trade[self.index - n]['type'] = 'low'
                                del self.hold[tic]
                            elif self.trade[self.index - n]['action'] == 'holding':
                                gain = round((stock['low'] - self.trade[self.index - n]['price']) *1000, 0)
                                if gain <= -1000:
                                    self.trade[self.index - n]['sell_date_detail'] = stock['date']
                                    self.trade[self.index - n]['sell_date'] = stock['date_only']
                                    self.trade[self.index - n]['sell_price'] = stock['close']
                                    self.trade[self.index - n]['gain'] = round((self.trade[self.index - n]['sell_price'] - self.trade[self.index - n]['price']) * 1000,0 )
                                    self.trade[self.index - n]['action'] = 'done'
                                    self.trade[self.index - n]['type'] = 'low'
                                    del self.hold[tic]
                            elif self.trade[self.index - n]['action'] == 'selling' and self.trade[self.index - n]['tic'] == tic:
                                self.trade[self.index - n]['sell_date_detail'] = stock['date']
                                self.trade[self.index - n]['sell_date'] = stock['date_only']
                                self.trade[self.index - n]['gain'] = round((self.trade[self.index - n]['sell_price'] - self.trade[self.index - n]['price']) * 1000,0) 
                                self.trade[self.index - n]['action'] = 'done'
                                del self.hold[tic]
                                self.pay_data = []
                                break
                        if len(self.pay_data) == 0:
                            break
                        self.sell_chose(stock, self.pay_data)
                    # print('trade',self.trade)
                #         if tic in self.pay_data:
                #             print('pay_data',self.pay_data[tic])
                #             print(tic,stock)
                # print('len trade',len(self.trade))
    def get_trade(self):
        sum = 0
        length = 0
        if self.trade:
            length = len(self.trade)
            for i in self.trade:
                sum = sum + i['gain']
        return sum,length

                               

    def pay_chose(self, stock , pay_data):
        if pay_data[stock['tic']] and stock['date'].time() < time(13,20):
            for pay in pay_data[stock['tic']]:
                if stock['high'] >= pay['pay'] and stock['low'] <= pay['pay']:
                    self.trade.append({'index': self.index,'tic': stock['tic'], 'action': 'buying', 'type': 'high', 'price': stock['close']})
                    self.index += 1
    
    def sell_chose(self, stock, pay_data):
        for n in range(1,self.chose_number+1):
            if self.trade[self.index - n]['action'] == 'holding' and self.trade[self.index - n]['tic'] == stock['tic']:
                if  stock['high'] >= pay_data[stock['tic']][0]['sell'] and stock['low'] <= pay_data[stock['tic']][0]['sell']:
                        self.trade[self.index - n]['action'] = 'selling'
                        self.trade[self.index - n]['sell_price'] = stock['close']
                

    def data_processing(self):
        """
        對已儲存的資料進行漲幅計算
        """
        processed_data = {}

        for tic, records in self.stock_data.items():
            if len(records) != 30:
                print(f"股票 {tic} 的資料不足 30 筆，無法處理。")
                continue

            # 將資料轉為 DataFrame 進行處理
            df = pd.DataFrame(records)

            # 計算漲幅
            reference_price = df.iloc[0]['open']
            for col in ['high', 'low', 'close', 'open']:
                df[col] = (((df[col] - reference_price) / reference_price)).round(10)

            processed_data[tic] = df

        # print("漲幅計算完成。")
        return processed_data

    def input_data_normalization(self, data):
        """
        根據給定的均值和標準差對數據進行歸一化
        """
        Input_open_mean = -0.00041714512936614356
        Input_open_std = 0.007071132962105002
        Input_high_mean = 0.0008918065658026626
        Input_high_std = 0.007192454354435754
        Input_low_mean = -0.001788884938551593
        Input_low_std = 0.007117076847735271
        Input_close_mean = -0.00044563458065097477
        Input_close_std = 0.007271157712421355
        Input_volume_mean = 299.12512601046126
        Input_volume_std = 960.7756073112927

        normalized_data = {}

        for tic, df in data.items():
            df['open'] = ((df['open'] - Input_open_mean) / Input_open_std).round(10)
            df['high'] = ((df['high'] - Input_high_mean) / Input_high_std).round(10)
            df['low'] = ((df['low'] - Input_low_mean) / Input_low_std).round(10)
            df['close'] = ((df['close'] - Input_close_mean) / Input_close_std).round(10)
            df['volume'] = ((df['volume'] - Input_volume_mean) / Input_volume_std).round(10)

            normalized_data[tic] = df

        # print("數據歸一化完成。")

        return normalized_data
    
    def input_data_totensor(self, data):
        """
        將歸一化後的數據轉換為 Tensor 並使用模型進行預測
        """
        results = {}

        for tic, df in data.items():
            # 將數據轉為 Tensor
            input_tensor = torch.tensor(df[['open', 'high', 'low', 'close', 'volume']].values, dtype=torch.float32)

            # 模型預測
            with torch.no_grad():
                output = self.model(input_tensor.unsqueeze(0))  # 模型接受 [batch_size, seq_len, features]
                results[tic] = output.squeeze(0).tolist()  # 將結果轉為 list

        # print("數據轉換為 Tensor 並完成預測。")
        return results
    
    def output_todata(self, model_output):
        """
        將模型輸出的數據進行還原，並處理大於 0.5 的機率作為分類標籤
        """
        Output_high_mean = 0.010313833666590584
        Output_high_std = 0.011032684949846954
        Output_low_mean = -0.011056402805920114
        Output_low_std = 0.010711948553928822
        Output_close_mean = -0.0005812611561911554
        Output_close_std = 0.015202675698237418

        restored_data = {}

        for tic, outputs in model_output.items():
            restored_data[tic] = []
            high = (outputs[0] * Output_high_std) + Output_high_mean
            low = (outputs[1] * Output_low_std) + Output_low_mean
            close = (outputs[2] * Output_close_std) + Output_close_mean
            large_volume = 1 if outputs[3] > 0.5 else 0

            restored_data[tic].append({
                'high': round(high, 10),
                'low': round(low, 10),
                'close': round(close, 10),
                'large_volume': large_volume
            })

        # print("模型輸出數據還原完成。")

        return restored_data
    
    def get_org_data(self, outputs):
        """
        將漲幅還原為實際售價
        """
        final_output = {}

        for tic, records in outputs.items():
            if tic not in self.stock_open:
                print(f"股票 {tic} 缺少開盤價數據，無法還原。")
                continue

            open_price = self.stock_open[tic]  # 每天每檔股票的第一筆開盤價
            final_output[tic] = []

            for record in records:
                high = round(record['high'] * open_price + open_price, 1)
                low = round(record['low'] * open_price + open_price, 1)
                close = round(record['close'] * open_price + open_price, 1)
                large_volume = record['large_volume']

                final_output[tic].append({
                    'high': high,
                    'low': low,
                    'close': close,
                    'large_volume': large_volume
                })

        # print("漲幅已還原為實際售價。")
        return final_output

    def get_indext(self, data):
        """
        計算每檔股票的指標：最高與最低波動、收盤與最低波動
        """
        index_data = {}

        for tic, records in data.items():
            index_data[tic] = []

            for record in records:
                high = record['high']
                low = record['low']
                close = record['close']
                large_volume = record['large_volume']

                high_low_volatility = round((high - low) / low, 4) if low != 0 else 0
                close_low_volatility = round((close - low) / low, 4) if low != 0 else 0

                index_data[tic].append({
                    'high': high,
                    'low': low,
                    'close': close,
                    'large_volume': large_volume,
                    'high_low_volatility': high_low_volatility,
                    'close_low_volatility': close_low_volatility
                })

        # print("指標計算完成。")
        return index_data
    
    def sort_with_indext(self, data):
        """
        根據指標對股票排序
        - 首先根據 large_volume 排序，1 在前
        - 其次根據 high_low_volatility 和 close_low_volatility 的幾何平均值排序
        """
        sorted_data = {}

        for tic, records in data.items():
            sorted_records = sorted(
                records,
                key=lambda x: (
                    -x['large_volume'],  # large_volume 降序
                    -(x['high_low_volatility'] * x['close_low_volatility']) ** 0.5  # 幾何平均降序
                )
            )
            sorted_data[tic] = sorted_records

        # print("排序完成。")
        return sorted_data
    
    def get_pay_number(self, data):
        count = 0
        pay_data = {}

        for tic, records in data.items():
            count = 0
            
            for record in records:
                high = record['high']
                low = record['low']
                close = record['close']
                high_low_volatility = record['high_low_volatility']
                close_low_volatility = record['close_low_volatility']
                if high_low_volatility > 0.02 and close_low_volatility > 0.012:
                    pay_data[tic] = []
                    pay = round((high + low) / 2 * 0.995, 2)
                    sell = round((high + low) / 2 * 1.001 ,2)
                   
                    pay_data[tic].append({
                        'pay': pay,
                        'sell': sell,
                    })
                    count += 1
                    if count == self.chose_number:
                        break
            if count == self.chose_number:
                break
        return pay_data
                




    

    

# 定義 LSTM 模型 v3 0.6420
class LSTM(nn.Module):
    def __init__(self, input_size=5, output_size=4):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True, bidirectional=True)  # 輸入大小改為 5
        self.mish = nn.Mish()
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)  # 輸出大小改為 output_size - 1 (前 3 個數值)
        self.fc = nn.Linear(128, 32)  # 輸出大小改為 output_size - 1 (前 3 個數值)
        self.fc4 = nn.Linear(32, output_size - 1)  # 輸出大小改為 output_size - 1 (前 3 個數值)
        self.fc_sigmoid = nn.Linear(32, 1)  # 單獨處理最後一個數值
        self.dropout = nn.Dropout(0.5)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()    

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data, nonlinearity='relu')
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.mish(x)
        x, _ = self.lstm2(x)
        x = self.mish(x)
        y = torch.mean(x, dim=1)
        x = self.fc1(y)
        x = self.prelu1(x)
        x = self.fc2(x)
        x = self.prelu2(x)
        x = self.fc3(x+y)
        x = self.prelu3(x)
        x = self.fc(x)
        if self.training:
            x = self.dropout(x)      
        # 分離兩部分輸出
        out_main = self.fc4(x)  # 前 3 個數值
        out_sigmoid = torch.sigmoid(self.fc_sigmoid(x))  # 單獨使用 Sigmoid 激活

        # 合併輸出
        output = torch.cat((out_main, out_sigmoid), dim=1)
        return output
