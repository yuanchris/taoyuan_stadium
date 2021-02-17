fin = open('out.txt', 'rt' )
past_number = -1
while True:
  try:
    line = fin.readline()
    if not line:
      break
    # print(line)
    if line == './out.txt\n' or line == '=== watchDog is running ===\n' \
    or line == './data1\n' or line == './data1':
      continue
    
    
    # get only number
    number = line[12:]
    number = number.split('.')[0]
    # print(number)
    
    if int(number) == past_number: # same profile: ignore
      continue
    if int(number) - past_number != 1:
      print(f'{past_number + 1} ~ {int(number) - 1} missed')
    # print(number)
    past_number = int(number)
  except:
    print(number)

fin.close()