import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sns.set_theme()
sns.set_context("paper")

model_dir = Path("./models/")
result_dir = Path("./result/")
fig_dir = Path("./paper/")
if not fig_dir.exists():
    fig_dir.mkdir()
def MSE(x, y):
    mse = np.sqrt(np.mean((x - y)**2))
    print(f"MSE = {mse}")
    return mse

def SSIM(x, y):
    k1 = 0.01
    k2 = 0.03
    vmax = np.max(x)
    vmin = np.min(x)
    μx = np.mean(x)
    μy = np.mean(y)
    σx = np.sqrt(np.mean((x-μx)**2))
    σy = np.sqrt(np.mean((y-μy)**2))
    σxy = np.mean((x-μx)*(y-μy))
    c1 = ( k1 * (vmax-vmin) )**2
    c2 = ( k2 * (vmax-vmin) )**2
    ssim = (2*μx*μy + c1) * (2*σxy + c2) / ((μx**2 + μy**2 + c1) * (σx**2 + σy**2 + c2))
#     print((2*μx*μy + c1), (μx**2 + μy**2 + c1),(2*σxy + c2), (σx**2 + σy**2 + c2))
    print(f"SSIM = {ssim}")
    return ssim

def PSNR(x, y):
    mse = np.sqrt(np.mean((x - y)**2))
    vmax = np.max(x)
    vmin = np.min(x)
    maxI = vmax - vmin
    psnr = 20.0 * np.log10(maxI) - 10.0 * np.log10(mse)
    print(f"PSNR = {psnr}")
    return psnr

def metrics(x, y):
    mse = MSE(x, y)
    ssim = SSIM(x, y)
    psnr = PSNR(x, y)

def plot_model(model, fig_name, fig_dir, p_vmax=None, p_vmin=None, s_vmax=None, s_vmin=None, cmap="jet"):
    x = np.arange(model['nx'][0][0]) * model['dx'][0][0]
    y = np.arange(model['ny'][0][0]) * model['dy'][0][0]
    #     plt.figure(figsize=(12,8))
    plt.figure()
    plt.subplot(221)
    x_mesh, y_mesh = np.meshgrid(x, y)
    x0 = x[0] / 1e3
    im = plt.pcolormesh(x_mesh / 1e3 - x0, y_mesh / 1e3, model['vp'].T / 1e3, vmax=p_vmax, vmin=p_vmin, rasterized=True,
                        shading='auto', cmap=cmap)
    plt.text(-0.1, 1.5, "(a)", ha='left', va='top', fontsize=10, transform=plt.gca().transAxes)
    plt.xlabel("X (km)")
    plt.ylabel("Z (km)")
    plt.gca().tick_params(top=True, left=True, labeltop=True, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax)
    cb.set_label('Vs(km/s)')
    cb.ax.tick_params(labelsize=7)

    plt.subplot(222)
    im = plt.pcolormesh(x_mesh / 1e3 - x0, y_mesh / 1e3, model['init_vp'].T / 1e3, vmax=p_vmax, vmin=p_vmin,
                        rasterized=True, shading='auto', cmap=cmap)
    plt.text(-0.1, 1.5, "(b)", ha='left', va='top', fontsize=10, transform=plt.gca().transAxes)
    plt.xlabel("X (km)")
    plt.gca().tick_params(top=True, left=True, labeltop=True, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax)
    cb.set_label('Vp(km/s)')
    cb.ax.tick_params(labelsize=7)

    plt.subplot(223)
    im = plt.pcolormesh(x_mesh / 1e3 - x0, y_mesh / 1e3, model['vs'].T / 1e3, vmax=s_vmax, vmin=s_vmin, rasterized=True,
                        shading='auto', cmap=cmap)
    plt.text(-0.1, 1.5, "(c)", ha='left', va='top', fontsize=10, transform=plt.gca().transAxes)
    plt.xlabel("X (km)")
    plt.ylabel("Z (km)")
    plt.gca().tick_params(top=True, left=True, labeltop=True, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax)
    cb.set_label('Vs(km/s)')
    cb.ax.tick_params(labelsize=7)

    plt.subplot(224)
    im = plt.pcolormesh(x_mesh / 1e3 - x0, y_mesh / 1e3, model['init_vs'].T / 1e3, vmax=s_vmax, vmin=s_vmin,
                        rasterized=True, shading='auto', cmap=cmap)
    plt.text(-0.1, 1.5, "(d)", ha='left', va='top', fontsize=10, transform=plt.gca().transAxes)
    plt.xlabel("X (km)")
    plt.gca().tick_params(top=True, left=True, labeltop=True, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax)
    cb.set_label('Vs(km/s)')
    cb.ax.tick_params(labelsize=7)
    plt.tight_layout()
    #     plt.savefig(fig_dir.joinpath(fig_name+".png"), bbox_inches='tight', dpi=300)
    plt.savefig(fig_dir.joinpath(fig_name + ".png"), bbox_inches='tight', dpi=300)
    plt.show()


def plot_results(model, result_nn, result_BFGS, fig_name, fig_dir, p_vmax=None, s_vmax=None, p_vmin=None, s_vmin=None,
                 cmap="jet"):
    x = np.arange(model['nx'][0][0]) * model['dx'][0][0]
    y = np.arange(model['ny'][0][0]) * model['dy'][0][0]
    idx = np.arange(0, model['nx'][0][0], int(np.floor(model['nx'][0][0]/5)))[1:-1]
    # plt.figure(figsize=(5,8))
    plt.figure()
    plt.subplot(221)
    x_mesh, y_mesh = np.meshgrid(x, y)
    x0 = x[0] / 1e3
    im = plt.pcolormesh(x_mesh / 1e3 - x0, y_mesh / 1e3, result_nn['cp'] / 1e3, vmax=p_vmax, vmin=p_vmin,
                        rasterized=True, shading='auto', cmap=cmap)
    for i in range(len(idx)):
        plt.plot([x[idx[i]]/1e3, x[idx[i]]/1e3], [y[0]/1e3, y[-1]/1e3], "k", linewidth=0.6)
    plt.text(-0.1, 1.5, "(a)", ha='left', va='top', fontsize=10, transform=plt.gca().transAxes)
    plt.xlabel("X (km)")
    plt.ylabel("Z (km)")
    plt.gca().tick_params(top=True, left=True, labeltop=True, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax)
    cb.set_label('Vs(km/s)')
    cb.ax.tick_params(labelsize=7)

    plt.subplot(222)
    im = plt.pcolormesh(x_mesh / 1e3 - x0, y_mesh / 1e3, result_BFGS['cp'] / 1e3, vmax=p_vmax, vmin=p_vmin,
                        rasterized=True, shading='auto', cmap=cmap)
    for i in range(len(idx)):
        plt.plot([x[idx[i]]/1e3, x[idx[i]]/1e3], [y[0]/1e3, y[-1]/1e3], "k", linewidth=0.6)
    plt.text(-0.1, 1.5, "(b)", ha='left', va='top', fontsize=10, transform=plt.gca().transAxes)
    plt.xlabel("X (km)")
    plt.gca().tick_params(top=True, left=True, labeltop=True, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax)
    cb.set_label('Vs(km/s)')
    cb.ax.tick_params(labelsize=7)


    plt.subplot(223)
    im = plt.pcolormesh(x_mesh / 1e3 - x0, y_mesh / 1e3, result_nn['cs'] / 1e3, vmax=s_vmax, vmin=s_vmin,
                        rasterized=True, shading='auto', cmap=cmap)
    for i in range(len(idx)):
        plt.plot([x[idx[i]]/1e3, x[idx[i]]/1e3], [y[0]/1e3, y[-1]/1e3], "k", linewidth=0.6)
    plt.text(-0.1, 1.5, "(c)", ha='left', va='top', fontsize=10, transform=plt.gca().transAxes)
    plt.xlabel("X (km)")
    plt.ylabel("Z (km)")
    plt.gca().tick_params(top=True, left=True, labeltop=True, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax)
    cb.set_label('Vs(km/s)')
    cb.ax.tick_params(labelsize=7)

    plt.subplot(224)
    im = plt.pcolormesh(x_mesh / 1e3 - x0, y_mesh / 1e3, result_BFGS['cs'] / 1e3, vmax=s_vmax, vmin=s_vmin,
                        rasterized=True, shading='auto', cmap=cmap)
    plt.text(-0.1, 1.5, "(d)", ha='left', va='top', fontsize=10, transform=plt.gca().transAxes)
    plt.xlabel("X (km)")
    plt.gca().tick_params(top=True, left=True, labeltop=True, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax)
    cb.set_label('Vs(km/s)')
    cb.ax.tick_params(labelsize=7)

    print("nn_vp")
    metrics(model['vp'], result_nn['cp'].T)
    print("nn_vs")
    metrics(model['vs'], result_nn['cs'].T)
    print("bfgs_vp")
    metrics(model['vp'], result_BFGS['cp'].T)
    print("bfgs_vs")
    metrics(model['vs'], result_BFGS['cs'].T)

    # plt.subplot(325)
    # im = plt.pcolormesh(x_mesh/1e3-x0, y_mesh/1e3, result_nn['rho']/1e3, vmax=s_vmax, vmin=s_vmin, rasterized=True, shading='auto', cmap=cmap)
    # plt.text(-0.1, 1.5, "(c)", ha='left', va='top', fontsize=10, transform = plt.gca().transAxes)
    # plt.xlabel("X (km)")
    # plt.ylabel("Z (km)")
    # plt.gca().tick_params(top=True, left=True, labeltop=True, labelbottom=False)
    # plt.gca().xaxis.set_label_position('top')
    # plt.gca().invert_yaxis()
    # plt.axis('scaled')

    # plt.subplot(326)
    # im = plt.pcolormesh(x_mesh/1e3-x0, y_mesh/1e3, result_BFGS['cs']/1e3, vmax=s_vmax, vmin=s_vmin, rasterized=True, shading='auto', cmap=cmap)
    # plt.text(-0.1, 1.5, "(d)", ha='left', va='top', fontsize=10, transform = plt.gca().transAxes)
    # plt.xlabel("X (km)")
    # plt.gca().tick_params(top=True, left=True, labeltop=True, labelbottom=False)
    # plt.gca().xaxis.set_label_position('top')
    # plt.gca().invert_yaxis()
    # plt.axis('scaled')
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # cb = plt.colorbar(im, cax)
    # cb.ax.set_title('Vs(km/s)')
    # cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(fig_dir.joinpath(fig_name + ".png"), bbox_inches='tight', dpi=300)
    plt.show()

    numbers = ["(i)", "(ii)", "(iii)", "(iv)", "(v)", "(vi)"]
    #     plt.figure(figsize=(10,4))
    plt.figure()
    for i in range(len(idx)):
        plt.subplot(1, len(idx), i + 1)
        plt.text(0.98, 0.98, f'{numbers[i]}', ha='right', va='top', fontsize="x-small", transform=plt.gca().transAxes)
        plt.plot(model["vp"][idx[i], :] / 1e3, y / 1e3, label="True")
        plt.plot(model["init_vp"][idx[i], :] / 1e3, y / 1e3, label="Initial model")
        plt.plot(result_nn['cp'].T[idx[i], :] / 1e3, y / 1e3, label="CNN-EFWI")
        plt.plot(result_BFGS['cp'].T[idx[i], :] / 1e3, y / 1e3, label="BFGS-EFWI")

        if i >= 1:
            plt.gca().set_yticklabels([])
        else:
            plt.ylabel("Z (km)")
        if i == len(idx) - 1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize="x-small")
        plt.gca().invert_yaxis()
        plt.autoscale(enable=True, axis='y', tight=True)
    plt.gcf().text(0.5, 0., 'P-wave Velocity (km/s)', ha='center', va="top", fontsize=11)
    plt.savefig(fig_dir.joinpath(fig_name + "vp_slice.png"), bbox_inches='tight', dpi=300)
    plt.show()

    plt.figure()
    for i in range(len(idx)):
        plt.subplot(1, len(idx), i + 1)
        plt.text(0.98, 0.98, f'{numbers[i]}', ha='right', va='top', fontsize="x-small", transform=plt.gca().transAxes)
        plt.plot(model["vs"][idx[i], :] / 1e3, y / 1e3, label="True")
        plt.plot(model["init_vs"][idx[i], :] / 1e3, y / 1e3, label="Initial model")
        plt.plot(result_nn['cs'].T[idx[i], :] / 1e3, y / 1e3, label="CNN-EFWI")
        plt.plot(result_BFGS['cs'].T[idx[i], :] / 1e3, y / 1e3, label="BFGS-EFWI")

        if i >= 1:
            plt.gca().set_yticklabels([])
        else:
            plt.ylabel("Z (km)")
        if i == len(idx) - 1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize="x-small")
        plt.gca().invert_yaxis()
        plt.autoscale(enable=True, axis='y', tight=True)
    plt.gcf().text(0.5, 0., 'S-wave Velocity (km/s)', ha='center', va="top", fontsize=11)
    plt.savefig(fig_dir.joinpath(fig_name + "vs_slice.png"), bbox_inches='tight', dpi=300)
    plt.show()


model_true = sio.loadmat(model_dir.joinpath("over_thrust_model.mat"))
plot_model(model_true, "true_and_initial_model", fig_dir,  p_vmax=5.5, p_vmin=1.5, s_vmax=3.5, s_vmin=0, cmap="rainbow")

result_nn = sio.loadmat("results/result100.mat")
result_BFGS = sio.loadmat("default/Results/cp142.mat")

plot_results(model_true, result_nn, result_BFGS, "result_model", fig_dir, p_vmax=5.5, p_vmin=1.5, s_vmax=3.5, s_vmin=0, cmap="rainbow")