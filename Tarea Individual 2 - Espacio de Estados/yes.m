function nyquist_points(k, w_range)
    % Function to compute Nyquist points for manual plotting
    % k: Gain value
    % w_range: Vector of frequency values (omega) to evaluate G(jw)

    % Define symbolic transfer function
    syms s
    Gs = (1) / (s^3 + 0.2 * s^2 + 1); % Open-loop transfer function
    
    % Substitute s = jω for each ω in w_range
    fprintf('   ω     Real(G(jω))   Imag(G(jω))\n');
    fprintf('----------------------------------\n');
    
    for w = w_range
        G_eval = subs(Gs, s, 1j * w); % Evaluate G(jω)
        real_part = double(real(G_eval));
        imag_part = double(imag(G_eval));
        
        % Print values in a table format
        fprintf('%6.2f   %10.4f   %10.4f\n', w, real_part, imag_part);
    end
end

k = 1; % Define the gain
w_values = linspace(0, 5, 21); % 20 frequency points from 0 to 10
nyquist_points(k, w_values);

