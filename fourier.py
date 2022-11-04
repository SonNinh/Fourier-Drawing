from typing import List
import math
import numpy as np
import cv2



class Fourier_serie:
    def __init__(self, 
                Sx: np.ndarray, # signal
                P: int, # period
                ) -> None:

        self.Sx = Sx
        self.SX_est = np.empty_like(Sx)
        self.vectors = None
        self.P = P
        self.N = 1
        self.Cn = [0] * len(self.Sx)


    def decompose(self, N: int) -> None:
        '''
        Calculate coefficients
        '''
        assert N > 0, "Number of harmony must be greater than 0"
        Ns = np.zeros((2*N+1, ), dtype=np.int16)
        Ns[1::2] = np.array(np.arange(1, N+1))
        Ns[2::2] = -np.array(np.arange(1, N+1))
        Ns = Ns.reshape(-1, 1)
        ts = np.array(np.arange(self.P))

        self.N = N 
        self.Cn = np.mean(
            self.Sx * np.exp(-1j*2*Ns*np.pi*ts/self.P), 
            axis=1, keepdims=True
        ) # (2N+1, 1)

    def inference(self, ):
        '''
        Estimate signal using Fourier serie
        '''
        ts = np.array(np.arange(self.P))
        Ns = np.zeros((2*self.N+1, ), dtype=np.int16)
        Ns[1::2] = np.array(np.arange(1, self.N+1))
        Ns[2::2] = -np.array(np.arange(1, self.N+1))
        Ns = Ns.reshape(-1, 1)

        self.vectors = self.Cn * np.exp(1j*2*Ns*np.pi*ts/self.P)
        self.SX_est = np.sum(
            self.vectors,
            axis=0,
            keepdims=False
        )# (2N+1, 1)
            


    def draw_path(self, img, xs, ys, t_valid, thickness, color, window_name) -> None:
        # draw original signal
        pre_x = xs[-1]
        pre_y = ys[-1]
        for x, y, is_valid in zip(xs, ys, t_valid):
            if is_valid:
                cv2.line(
                    img,
                    (int(x), int(y)),
                    (int(pre_x), int(pre_y)),
                    color=color,
                    thickness=thickness
                )
                cv2.imshow(window_name, img)
                cv2.waitKey(2)

            pre_x = x
            pre_y = y


    def draw(self, t_valid: List) -> np.ndarray:

        # output parameters such as fps and size can be changed

        window_name = 'Fourier drawing'
        cv2.namedWindow(window_name)
        xs = self.Sx.real
        ys = self.Sx.imag

        img_width = int(np.max(xs) + np.min(xs))
        img_height = int(np.max(ys) + np.min(ys))
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # output = cv2.VideoWriter("output.mp4", fourcc, 60, (img_width, img_height))

        while 1:
            img = np.zeros((img_height, img_width, 3), np.uint8)
            line_thickness = max(3, min(img_width, img_height) // 150)
            # self.draw_path(
            #     img, self.Sx.real, self.Sx.imag, t_valid, 
            #     line_thickness*2, (0, 0, 255), window_name
            # )
            # self.draw_path(
            #     img, self.SX_est.real[::-1], self.SX_est.imag[::-1], t_valid[::-1], 
            #     line_thickness, (255, 0, 0), window_name
            # )

            # list_img = []
            for t in np.arange(self.P):
                img_cp = np.copy(img)
                pre_x = self.vectors[0, t].real
                pre_y = self.vectors[0, t].imag
                for n in range(1, self.N*2+1):
                    x = self.vectors[n, t].real + pre_x
                    y = self.vectors[n, t].imag + pre_y
                    cv2.line(
                        img_cp,
                        (int(x), int(y)),
                        (int(pre_x), int(pre_y)),
                        color=(255, 255, 255),
                        thickness=1
                    )

                    radius = int(math.sqrt(self.vectors[n, t].real**2 + self.vectors[n, t].imag**2))
                    if radius > 10:
                        cv2.circle(img_cp, (int(pre_x), int(pre_y)), 2, (0, 255, 0), thickness=-1)
                        cv2.circle(img_cp, (int(pre_x), int(pre_y)), radius, (150, 150, 150), thickness=1)
                    
                    pre_x = x
                    pre_y = y
                
                if t_valid[t] and n > 0:
                    cv2.circle(img, (int(pre_x), int(pre_y)), line_thickness//2, (200, 0, 255), thickness=-1)
                
                # output.write(img_cp)
                cv2.imshow(window_name, img_cp)
                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    # output.release()
                    cv2.destroyAllWindows()
                    return

            # break
        
        # imageio.mimsave('cat.gif', list_img, fps=100)
        

    