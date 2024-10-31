import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit,
    QGridLayout, QMessageBox
)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ProjectileMotionSimulation(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Моделирование движения тела с сопротивлением воздуха')
        self.setGeometry(100, 100, 900, 700)

        self.layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        double_validator_positive = QDoubleValidator(0.0001, 1e6, 4)
        double_validator_non_negative = QDoubleValidator(0.0, 1e6, 4)
        double_validator_angle = QDoubleValidator(0.0, 90.0, 2)

        self.v0_input = QLineEdit()
        self.v0_input.setValidator(double_validator_positive)
        self.v0_input.setPlaceholderText("Например: 50")
        self.grid_layout.addWidget(QLabel('Начальная скорость (м/с):'), 0, 0)
        self.grid_layout.addWidget(self.v0_input, 0, 1)

        self.angle_input = QLineEdit()
        self.angle_input.setValidator(double_validator_angle)
        self.angle_input.setPlaceholderText("0 - 90 градусов")
        self.grid_layout.addWidget(QLabel('Угол броска (градусы):'), 1, 0)
        self.grid_layout.addWidget(self.angle_input, 1, 1)

        self.height_input = QLineEdit()
        self.height_input.setValidator(double_validator_non_negative)
        self.height_input.setPlaceholderText("Например: 10")
        self.grid_layout.addWidget(QLabel('Начальная высота (м):'), 2, 0)
        self.grid_layout.addWidget(self.height_input, 2, 1)

        self.k_input = QLineEdit()
        self.k_input.setValidator(double_validator_non_negative)
        self.k_input.setPlaceholderText("Например: 0,1")
        self.grid_layout.addWidget(QLabel('Коэффициент сопротивления среды k (1/с):'), 3, 0)
        self.grid_layout.addWidget(self.k_input, 3, 1)

        self.layout.addLayout(self.grid_layout)

        self.start_button = QPushButton('Запустить моделирование')
        self.start_button.clicked.connect(self.start_simulation)
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        self.figure, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)

    def start_simulation(self):
        try:
            v0_text = self.v0_input.text().replace(',', '.')
            angle_text = self.angle_input.text().replace(',', '.')
            h0_text = self.height_input.text().replace(',', '.')
            k_text = self.k_input.text().replace(',', '.')

            if not v0_text or not angle_text or not h0_text or not k_text:
                raise ValueError("Все поля должны быть заполнены.")

            v0 = float(v0_text)
            angle = float(angle_text)
            h0 = float(h0_text)
            k = float(k_text)

            if v0 <= 0:
                raise ValueError("Начальная скорость должна быть положительной.")
            if not (0 <= angle <= 90):
                raise ValueError("Угол броска должен быть в диапазоне от 0 до 90 градусов.")
            if h0 < 0:
                raise ValueError("Начальная высота не может быть отрицательной.")
            if k < 0:
                raise ValueError("Коэффициент сопротивления k должен быть неотрицательным.")

        except ValueError as e:
            self.show_error(str(e))
            return

        theta = np.radians(angle)
        x0 = 0.0
        y0 = h0
        vx0 = v0 * np.cos(theta)
        vy0 = v0 * np.sin(theta)

        t0 = 0.0
        t_end = 100.0
        dt = 0.01

        t = [t0]
        x = [x0]
        y = [y0]
        vx = [vx0]
        vy = [vy0]
        v = [v0]

        g = 9.81

        while y[-1] >= 0 and t[-1] <= t_end:
            xi, yi = x[-1], y[-1]
            vxi, vyi = vx[-1], vy[-1]
            ti = t[-1]

            def dvx_dt(vx):
                return -k * vx

            def dvy_dt(vy):
                return -g - k * vy

            def dx_dt(vx):
                return vx

            def dy_dt(vy):
                return vy

            k1_vx = dvx_dt(vxi)
            k1_vy = dvy_dt(vyi)
            k1_x = dx_dt(vxi)
            k1_y = dy_dt(vyi)

            k2_vx = dvx_dt(vxi + 0.5 * dt * k1_vx)
            k2_vy = dvy_dt(vyi + 0.5 * dt * k1_vy)
            k2_x = dx_dt(vxi + 0.5 * dt * k1_vx)
            k2_y = dy_dt(vyi + 0.5 * dt * k1_vy)

            k3_vx = dvx_dt(vxi + 0.5 * dt * k2_vx)
            k3_vy = dvy_dt(vyi + 0.5 * dt * k2_vy)
            k3_x = dx_dt(vxi + 0.5 * dt * k2_vx)
            k3_y = dy_dt(vyi + 0.5 * dt * k2_vy)

            k4_vx = dvx_dt(vxi + dt * k3_vx)
            k4_vy = dvy_dt(vyi + dt * k3_vy)
            k4_x = dx_dt(vxi + dt * k3_vx)
            k4_y = dy_dt(vyi + dt * k3_vy)

            vxi_next = vxi + (dt / 6.0) * (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx)
            vyi_next = vyi + (dt / 6.0) * (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy)

            xi_next = xi + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            yi_next = yi + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

            ti_next = ti + dt

            vx.append(vxi_next)
            vy.append(vyi_next)
            x.append(xi_next)
            y.append(yi_next)
            t.append(ti_next)
            v.append(np.sqrt(vxi_next ** 2 + vyi_next ** 2))

            if yi_next < 0:
                break

        for ax in self.axs.flat:
            ax.clear()
            ax.grid(True)

        colors = ['blue', 'green', 'red', 'magenta']

        self.axs[0, 0].plot(x, y, color=colors[0], label='Траектория')
        self.axs[0, 0].set_xlabel('X (м)')
        self.axs[0, 0].set_ylabel('Y (м)')
        self.axs[0, 0].set_title('Траектория движения тела')
        self.axs[0, 0].legend()
        self.axs[0, 0].grid(True)

        self.axs[0, 1].plot(t, x, color=colors[1], label='X(t)')
        self.axs[0, 1].set_xlabel('Время (с)')
        self.axs[0, 1].set_ylabel('X (м)')
        self.axs[0, 1].set_title('Координата X от времени')
        self.axs[0, 1].legend()
        self.axs[0, 1].grid(True)

        self.axs[1, 0].plot(t, y, color=colors[2], label='Y(t)')
        self.axs[1, 0].set_xlabel('Время (с)')
        self.axs[1, 0].set_ylabel('Y (м)')
        self.axs[1, 0].set_title('Координата Y от времени')
        self.axs[1, 0].legend()
        self.axs[1, 0].grid(True)

        self.axs[1, 1].plot(t, v, color=colors[3], label='v(t)')
        self.axs[1, 1].set_xlabel('Время (с)')
        self.axs[1, 1].set_ylabel('Скорость (м/с)')
        self.axs[1, 1].set_title('Скорость от времени')
        self.axs[1, 1].legend()
        self.axs[1, 1].grid(True)

        for ax in self.axs.flat:
            ax.tick_params(direction='in', which='both', top=True, right=True)

        self.figure.tight_layout()
        self.canvas.draw()

    def show_error(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle('Ошибка ввода')
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


def main():
    app = QApplication(sys.argv)
    window = ProjectileMotionSimulation()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
