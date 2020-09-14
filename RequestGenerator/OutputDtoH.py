import sys
import os


def pull_subset(Req_name, Result_path, algo):
	if os.path.isdir(Result_path) == False:
		os.system("mkdir " + Result_path)

	algo_path = Result_path + "/" + Req_name + algo + "_O"

	os.system("mkdir " + algo_path)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+algo+"_O/ "+algo_path)

def pull(Req_name, Result_path):
	
	if os.path.isdir(Result_path) == False:
		os.system("mkdir " + Result_path)
		
	gpu = Result_path + "/" + Req_name + "gpu_O"
	dsp = Result_path + "/" + Req_name + "dsp_O"
	lb = Result_path + "/" + Req_name + "lb_O"
	st = Result_path + "/" + Req_name + "st_O"
	my = Result_path + "/" + Req_name + "my_O"
	slo = Result_path + "/" + Req_name + "slo_O"
	slo_div = Result_path + "/" + Req_name + "slo_div_O"
		
	os.system("mkdir " + gpu)
	os.system("mkdir " + dsp)
	os.system("mkdir " + lb)
	os.system("mkdir " + st)
	os.system("mkdir " + my)
	os.system("mkdir " + slo)
	os.system("mkdir " + slo_div)
	
	
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"gpu_O/ " +gpu)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"dsp_O/ " +dsp)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"lb_O/ " +lb)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"st_O/ " +st)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"my_O/ " +my)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"slo_O/ " +slo)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"slo_div_O/ " +slo_div)


if __name__=="__main__":
	parent_path = "/home/wonik/Downloads/snpe-1.25.1.310/exper_result/ATC20/REAL_result/"

	print("Usage: pull.py <remote request_name> <local result_path>")  
	print("Example Usage: pull.py poisson_avlg poisson_avlg (algo)")  
	Req_name = sys.argv[1]	
	Result_path = sys.argv[2]
	
	algo = ""

	if len(sys.argv) == 4:
		algo = sys.argv[3]
		Result_path = parent_path + Result_path		
		pull_subset(Req_name, Result_path, algo)

	if len(sys.argv) == 3:
		Result_path = parent_path + Result_path		
		print("Request_name: ", Req_name)
		print("Request_name: ", Result_path)
	
		pull(Req_name, Result_path)




