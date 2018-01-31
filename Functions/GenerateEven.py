import sys

def csv_generate_even(fout,min,max,distance=0.05):
    """
    Generate an evenly distributed CSV
    """
    x_min, y_min, z_min = min
    x_max, y_max, z_max = max

    with open(fout, "w+") as fl:
        for i in range(x_min, x_max):
            for id in range(0, int(1/distance)):
                for j in range(y_min, y_max):
                    for jd in range(0, int(1/distance)):
                        for k in range(z_min, z_max):
                            for kd in range(0, int(1/distance)):
                                print(i + id * distance)
                                fl.write(
                                    str(i + id * distance) +
                                    "," +
                                    str(j + jd * distance) +
                                    "," +
                                    str(k + kd * distance) +
                                    ",0,10,0\n"
                                    )
                    # generate stepping

    fl.close()
                


if __name__ == '__main__':
    min = (0,0,0)
    max = (100,100,100)
    csv_generate_even(sys.argv[1], min, max)