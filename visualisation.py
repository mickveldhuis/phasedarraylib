import numpy as np
import matplotlib.pyplot as plt

from phasedarraylib.utilities import decibel

def plot_configuration(x, y, aspect='equal'):
        fig, frame = plt.subplots(figsize=(4, 4))

        frame.scatter(x, y, color='black')

        frame.update(dict(
            xlabel=r'$x$ (m)',
            ylabel=r'$y$ (m)',
            aspect=aspect,
        ))

        fig.tight_layout()
        plt.show()


def plot_array_factor_1d(array_factor, u, freq, uv=0, axis='u', db_threshold=-40):
    fig, frame = plt.subplots(figsize=(7, 5))

    frame.plot(u, decibel(array_factor), lw=1, color='black')

    uv_alt = 'v' if axis == 'u' else 'u'

    frame.update(dict(
        title='{} MHz / ${}={}$'.format(freq, uv_alt, uv),
        xlabel=r'${}$'.format(axis),
        ylabel=r'$|\mathrm{AF}|$ (dB)',
        xlim=(-1, 1),
        xticks=np.arange(-1, 1.5, 0.5),
        ylim=(db_threshold, 0)
    ))

    frame.grid(alpha=0.2)

    fig.tight_layout()
    plt.show()

    return


def plot_array_factor_2d(array_factor, freq, db_threshold=-40, xframes=3, fn=''):
    if isinstance(freq, list | tuple | np.ndarray):
        xf = 1
        
        if freq.size > xframes:
            xf = freq.size % xframes
        
        fig, frame = plt.subplots(xf, 3, figsize=(5 * freq.size, 5), sharey=True, gridspec_kw=dict(wspace=0.16))
        
        frame = frame.ravel()
        
        for idx, f in enumerate(freq):
            data = decibel(array_factor[idx])
            image = frame[idx].imshow(data, origin='lower', vmin=db_threshold, vmax=0, extent=(-1, 1, -1, 1))
            
            frame[idx].update(dict(
                title='{} MHz'.format(f),
                xlabel=r'$u$',
                ylabel=r'$v$',
                xticks=np.linspace(-1, 1, 5),
                yticks=np.linspace(-1, 1, 5)
            ))
            
            if idx > 0:
                frame[idx].update(dict(ylabel=''))
        
        if 3*xf > freq.size:
            for idx in range(freq.size, 3*xf):
                frame[idx].axis('off')
        
    else:
        fig, frame = plt.subplots(figsize=(5, 5), sharey=True, gridspec_kw=dict(wspace=0.16))

        data = decibel(array_factor)
        image= frame.imshow(data, origin='lower', vmin=db_threshold, vmax=0, extent=(-1, 1, -1, 1))

        frame.update(dict(
            title='{} MHz'.format(freq),
            xlabel=r'$u$',
            ylabel=r'$v$',
            xticks=np.linspace(-1, 1, 5),
            yticks=np.linspace(-1, 1, 5)
        ))

    cb_frame = fig.add_axes([0.92, 0.15, 0.04, 0.7])
    colorbar = fig.colorbar(image, cax=cb_frame, label='$|\mathrm{AF}|$ (dB)')
    
    if fn:
        # TODO: make images folder if it doesn't exists!
        fig.savefig('images/{}.png'.format(fn), bbox_inches='tight')

    plt.show()