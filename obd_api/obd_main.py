import obd

connection = obd.OBD() # usb 또는 rf 포트에 자동으로 연결

fuel_rate = connection.query(obd.commands.FUEL_RATE ) # 연료율 구하기

if fuel_rate.is_null(): # 만약 fuel_rate가 null이라면
     print("NO data\n")
else:
     print(fuel_rate.value)
     