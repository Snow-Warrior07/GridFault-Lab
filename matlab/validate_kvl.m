% Independently check two-ended location algebra in MATLAB, not EMT simulation.
% Run from repository root: matlab -batch "run('matlab/validate_kvl.m')"
repoRoot = fileparts(fileparts(mfilename('fullpath')));
cases = jsondecode(fileread(fullfile(repoRoot,'tests','fixtures','matlab_cases.json')));
errors = zeros(numel(cases),1);
residuals = errors;
for k=1:numel(cases)
    p=cases(k).record.phasors;
    vs=[complex(p.s_va.re,p.s_va.im);complex(p.s_vb.re,p.s_vb.im);complex(p.s_vc.re,p.s_vc.im)];
    vr=[complex(p.r_va.re,p.r_va.im);complex(p.r_vb.re,p.r_vb.im);complex(p.r_vc.re,p.r_vc.im)];
    is=[complex(p.s_ia.re,p.s_ia.im);complex(p.s_ib.re,p.s_ib.im);complex(p.s_ic.re,p.s_ic.im)];
    ir=[complex(p.r_ia.re,p.r_ia.im);complex(p.r_ib.re,p.r_ib.im);complex(p.r_ic.re,p.r_ic.im)];
    z=complex(cases(k).z_re,cases(k).z_im);
    a=z*(is+ir); b=vs-vr+z*ir;
    m=real(a'*b)/real(a'*a);
    errors(k)=abs(m-cases(k).truth_pu);
    residuals(k)=norm(b-m*a)/norm(b);
end
assert(max(errors)<1e-9,'KVL location check failed');
report=struct('cases',numel(cases),'max_location_error_pu',max(errors),'max_electrical_residual',max(residuals),'matlab_version',version,'scope','Independent algebra check only; not EMT or field validation');
fid=fopen(fullfile(repoRoot,'results','matlab_validation.json'),'w');
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));fclose(fid);
disp(report);

