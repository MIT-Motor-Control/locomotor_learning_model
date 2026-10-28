function comparison = validate_fall_fit(table_dir)
% Refit the Figure 3D exponential fall-time curves and check them against the source table.
% For each learning rate, y = a*exp(b*x) is fit by least squares, with a in
% closed form for each b and b found by bounded scalar search.
% Inputs:  table_dir  - folder with figure3_d_fit_inputs.csv and figure3_d_fit_coefficients.csv
% Outputs: comparison - fitted a, b and SSE per learning rate, with the maximum
%          curve difference and SSE difference from the reference fit; errors if
%          either exceeds its tolerance

inputs = readtable(fullfile(table_dir, 'figure3_d_fit_inputs.csv'));
reference = readtable(fullfile(table_dir, 'figure3_d_fit_coefficients.csv'));
rows = zeros(height(reference), 6);
for i = 1:height(reference)
    selected = inputs.learning_rate == reference.learning_rate(i) & isfinite(inputs.mean_strides_until_fall);
    x = inputs.delta_mps(selected);
    y = inputs.mean_strides_until_fall(selected);
    objective = @(b) sum((amplitude(x, y, b) .* exp(b .* x) - y).^2);
    b = fminbnd(objective, -30, 30, optimset('TolX', 1e-12, 'Display', 'off'));
    a = amplitude(x, y, b);
    sse = objective(b);
    expected = reference.a(i) .* exp(reference.b(i) .* x);
    max_difference = max(abs(a .* exp(b .* x) - expected));
    assert(abs(sse-reference.sse(i)) < 1e-8*max(1,sse), 'Least-squares objectives disagree.');
    assert(max_difference < 1e-4, 'Fitted curves disagree by more than 0.0001 strides.');
    rows(i,:) = [reference.learning_rate(i), a, b, sse, max_difference, abs(sse-reference.sse(i))];
end
comparison = array2table(rows, 'VariableNames', {'learning_rate', 'a', 'b', 'sse', 'max_curve_difference', 'sse_difference'});
disp(comparison);
end

function a = amplitude(x, y, b)
% Return the least-squares amplitude of y = a*exp(b*x) for fixed b.
basis = exp(b .* x);
a = (basis' * y) / (basis' * basis);
end
