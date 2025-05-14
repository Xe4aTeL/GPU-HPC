from random import randint, random

d_size = 256

# For prg1
print(f"d_i.txt")
with open(f"d_i.txt", "w") as f:
	f.write(str(randint(1, 5)))

for _ in range(7): # for 256 -> 16384 sizes
	print(d_size)

	# For prg1
	print(f"{d_size}_MB_i.txt")
	with open(f"{d_size}_MB_i.txt", "w") as f:
		for _ in range(d_size ** 2):
			f.write(f"{randint(1, 5)} ")

	print(f"{d_size}_MC_i.txt")
	with open(f"{d_size}_MC_i.txt", "w") as f:
		for _ in range(d_size ** 2):
			f.write(f"{randint(1, 5)} ")
	
	print(f"{d_size}_ME_i.txt")
	with open(f"{d_size}_ME_i.txt", "w") as f:
		for _ in range(d_size ** 2):
			f.write(f"{randint(1, 5)} ")
	
	# For prg2
	print(f"{d_size}_B_f.txt")
	with open(f"{d_size}_B_f.txt", "w") as f:
		for _ in range(d_size):
			f.write("{:f} ".format(random() * 10 % 5))
	
	print(f"{d_size}_MC_f.txt")
	with open(f"{d_size}_MC_f.txt", "w") as f:
		for _ in range(d_size ** 2):
			f.write("{:f} ".format(random() * 10 % 5))
	
	print(f"{d_size}_MD_f.txt")
	with open(f"{d_size}_MD_f.txt", "w") as f:
		for _ in range(d_size ** 2):
			f.write("{:f} ".format(random() * 10 % 5))
	
	print(f"{d_size}_E_f.txt")
	with open(f"{d_size}_E_f.txt", "w") as f:
		for _ in range(d_size):
			f.write("{:f} ".format(random() * 10 % 5))
	
	d_size *= 2
	