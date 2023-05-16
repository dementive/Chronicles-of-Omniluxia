

standard = """ color = rgb { 1 1 1 }
color2 = rgb { 1 1 1 }

ship_names = {
} """


file_name = input("Insert name: ")
how_many_times = input("Insert number of duplicates: ")

for i in range(int(how_many_times)):
	f = open(str(file_name)+"_"+str(i+1)+".txt","w")

	f.write(standard)

	f.close()

print("operation_compleated_sucesfully")