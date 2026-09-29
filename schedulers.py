
import numpy as np
import matplotlib.pyplot as plt
t = np.array([0.933, 0.866, 0.8, 0.733, 0.666, 0.6, 0.533, 0.466, 0.4, 0.333, 0.266, 0.2, 0.133, 0.066, 0.0])
i = range(len(t))


plt.figure(figsize=(9, 5.5))
plt.plot(i, t, 'o-', label=r"$t_i$", linewidth=2)
plt.xlabel(r"denoising step $i$")
plt.ylabel(r"$t_i$")
plt.title(r"$t_i$")
plt.xlim(0, len(t))
plt.ylim(-0.02, 1.02)
plt.grid(True, alpha=0.25)
plt.legend(ncol=2)
plt.tight_layout()

path = "t_i.png"
plt.savefig(path, dpi=180, bbox_inches="tight")
plt.show()
plt.close()


taus = range(2, 50, 2) 
plt.figure(figsize=(9, 5.5))
for tau in taus:
    print(f"tau: {tau}")
    blend_rate = 1 - t**tau
    print(f"t**tau: {(t**tau).astype(float).round(3)}")
    plt.plot(i, 1-blend_rate, 'o-', label=fr"$\rho={tau}$", linewidth=2)

plt.xlabel(r"denoising step $i$")
plt.ylabel(r"$W^{src}(t_i)$")
plt.title(r"$W^{src}(t_i)=t_i^\rho$")
plt.xlim(0, len(t))
plt.ylim(-0.02, 1.02)
plt.grid(True, alpha=0.25)
plt.legend(ncol=2)
plt.tight_layout()

path = "f_t_power_tau_comparison.png"
plt.savefig(path, dpi=180, bbox_inches="tight")
plt.show()
plt.close()




taus = [0.01, 0.15, 0.25, 0.33, 0.5, 0.75,1.0, 1.5]
plt.figure(figsize=(9, 5.5))
for tau in taus:
    x = np.clip(np.array(i) / ((len(t)-1)* tau), 0, 1)
    blend_rate = 1 - 0.5 * (1 + np.cos(np.pi * x))
    plt.plot(np.array(i), 1-blend_rate, 'o-', label=fr"$\rho={tau}$",linewidth=2)

plt.xlabel(r"denoising step $i$")
plt.ylabel(r"$W^{src}(t_i)$")
plt.title(r"$W^{src}(t_i)= 0.5 * (1 + cos (\pi * min( \frac{i}{N \rho} , 1.0) ) )$")
plt.xlim(0, len(t))
plt.ylim(-0.02, 1.02)
plt.grid(True, alpha=0.25)
plt.legend(ncol=2)
plt.tight_layout()

path = "f_t_cos_index_tau_comparison.png"
plt.savefig(path, dpi=180, bbox_inches="tight")
plt.show()
plt.close()




# # plt.figure(figsize=(9, 5.5))
# # for tau in taus:
# #     x = np.clip((1-t)/tau, 0., 1.)
# #     blend_rate = 1 - 0.5 * (1 + np.cos(np.pi * x))
# #     plt.plot(np.array(i), 1-blend_rate, 'o-', label=fr"$\rho={tau}$",linewidth=2)

# # plt.xlabel(r"denoising step $i$")
# # plt.ylabel(r"$W^{src}(t_i)$")
# # plt.title(r"$W^{src}(t_i)= 0.5 * (1 + cos (\pi * min( \frac{1-t_i}{\rho} , 1.0) ) )$")
# # plt.xlim(0, len(t))
# # plt.ylim(-0.02, 1.02)
# # plt.grid(True, alpha=0.25)
# # plt.legend(ncol=2)
# # plt.tight_layout()

# # path = "f_t_cos_time_tau_comparison.png"
# # plt.savefig(path, dpi=180, bbox_inches="tight")
# # plt.show()


plt.figure(figsize=(9, 5.5))
tau_min = [2.]
tau_max = [50.]
M = np.linspace(0, 1, 100)

for t_min, t_max in zip(tau_min, tau_max):
    print(f"t_min: {t_min}, t_max: {t_max}")
    A_min = 1/(1+t_max)
    A_max = 1/(1+t_min)
    tau = 1 / ((1-M) * A_max + M * A_min) - 1
    for i, m in enumerate(M):
        print(f"tau({m}): {tau[i]}")
    plt.plot(M, tau, 'o-', label=fr"$\rho min={t_min}, \rho max={t_max}$",linewidth=2)

plt.xlabel(r"edit measure $m$")
plt.ylabel(r"$tau(m)$")
plt.title(r"$tau(m) = \frac{1}{A(m)} - 1$")
plt.xlim(0, 1)
plt.ylim(tau_min[0], tau_max[-1])
plt.grid(True, alpha=0.25)
plt.legend(ncol=2)
plt.tight_layout()

path = "tau_m_comparison.png"
plt.savefig(path, dpi=180, bbox_inches="tight")
plt.show()
plt.close()
