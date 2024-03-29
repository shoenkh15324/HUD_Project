import obd

connection = obd.OBD() # usb 또는 rf 포트에 자동으로 연결

cmd_fuel_rate = obd.commands.FUEL_RATE
reponse_fuel_rate = connection.query(cmd_fuel_rate)

if reponse_fuel_rate.is_null():
     print("NO data\n")
else:
     print(reponse_fuel_rate.value)     

