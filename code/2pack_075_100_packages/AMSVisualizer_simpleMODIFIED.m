classdef AMSVisualizer_simpleMODIFIED < rl.env.viz.AbstractFigureVisualizer
    %AMSVISUALIZER_SIMPLE Fast visualizer for the AMS sorting environment.
    % watch out for the CLASS name
    properties (Access = private)
        EvalTopRecorded = []
        EvalOrder = []
        EvalGaps = []
        EvalSuccessCount = 0
        LastCont = -1
        
        % --- CACHE DEGLI HANDLE GRAFICI PER MASSIMA FLUIDITÀ ---
        ParcelRectHandles = []   % Handle dei rettangoli dei pacchi
        ParcelTextHandles = []   % Handle dei testi dei pacchi (numeri)
        InfoTextHandle1 = []     % Handle della prima riga di info
        InfoTextHandle2 = []     % Handle della seconda riga di info
        TreadmillTextHandle = [] % Handle del testo della velocità del nastro
    end
    methods
        function this = AMSVisualizer_simpleMODIFIED(env) % name change
            this = this@rl.env.viz.AbstractFigureVisualizer(env);
        end
    end
    methods (Access = protected)
        function f = buildFigure(this)
            f = figure( ...
                'Toolbar','none', ...
                'Visible','on', ...
                'HandleVisibility','off', ...
                'NumberTitle','off', ...
                'Name','AMS sorting visualizer simple', ...
                'CloseRequestFcn',@(~,~)delete(this));
            if ~strcmp(f.WindowStyle,'docked')
                f.Position = [220 80 900 720];
            end
            f.MenuBar = 'none';
            ha = axes('Parent',f);
            hold(ha,'on');
            axis(ha,'equal');
            box(ha,'on');
            grid(ha,'off');
            
            env = this.Environment;
            xMax = env.d_AMS*env.n_j_AMS;
            yExit = env.d_AMS*env.n_i_AMS;
            yEval = yExit + 1.0;
            dAMS = env.d_AMS;
            
            ha.XLim = [0 xMax];
            ha.YLim = [0 yEval];
            xlabel(ha,'x [m]');
            ylabel(ha,'y [m]');
            title(ha,'AMS sorting environment - simple view');
            
            % -----------------------------------------------------------------
            % 1. DISEGNO DEGLI ELEMENTI STATICI (ESEGUITO UNA SOLA VOLTA)
            % -----------------------------------------------------------------
            % Outfeed treadmill background
            rectangle(ha,'Position',[0 yExit xMax 1.0], ...
                'FaceColor',[0.92 0.92 0.92], 'EdgeColor','none');
            
            % AMS Background
            rectangle(ha,'Position',[0 0 xMax yExit], ...
                'FaceColor',[1 1 1], 'EdgeColor',[0 0 0], 'LineWidth',1.5);
            
            % AMS Grid Lines
            for ii = 0:env.n_i_AMS
                line(ha,[0 xMax],[ii*dAMS ii*dAMS], 'LineWidth',1.0, 'Color',[0 0 0]);
            end
            for jj = 0:env.n_j_AMS
                line(ha,[jj*dAMS jj*dAMS],[0 yExit], 'LineWidth',1.0, 'Color',[0 0 0]);
            end
            
            % AMS exit line
            line(ha,[0 xMax],[yExit yExit], 'LineWidth',2.5, 'LineStyle','--', 'Color',[0.20 0.20 0.20]);
            
            % Evaluation line
            line(ha,[0 xMax],[yEval yEval], 'LineWidth',3.0, 'LineStyle','-', 'Color',[0.85 0.10 0.10]);
            
            % Testi fissi
            text(ha,0.02,yExit+0.03,'AMS exit', 'FontSize',9, 'FontWeight','bold', 'Color',[0.20 0.20 0.20]);
            text(ha,0.02,yEval-0.05,'accuracy evaluation line: 1 m after AMS exit', 'FontSize',9, 'FontWeight','bold', 'Color',[0.85 0.10 0.10]);
            
            % Frecce direzione nastro statiche
            for xx = linspace(0.20,xMax-0.20,3)
                quiver(ha,xx,yExit+0.15,0,0.16,0, 'LineWidth',1.0, 'MaxHeadSize',1.5, 'Color',[0.35 0.35 0.35]);
            end
            
            % -----------------------------------------------------------------
            % 2. PRE-ALLOCAZIONE DEGLI ELEMENTI DINAMICI (VUOTI O INVISIBILI)
            % -----------------------------------------------------------------
            % Testo dinamico del treadmill
            this.TreadmillTextHandle = text(ha,0.02,yExit+0.50,'', 'FontSize',9, 'Color',[0.25 0.25 0.25]);
            
            % Pannelli informativi di testo in basso
            this.InfoTextHandle1 = text(ha,0.02,yEval-0.18,'', 'FontSize',10, 'FontWeight','bold', 'BackgroundColor',[1 1 1], 'Margin',3);
            this.InfoTextHandle2 = text(ha,0.02,yEval-0.28,'', 'FontSize',10, 'FontWeight','bold', 'BackgroundColor',[1 1 1], 'Margin',3);
            
            % Generazione preliminare di rettangoli e scritte per i pacchi usando i Placeholder
            maxBoxes = env.max_gen_boxes;
            this.ParcelRectHandles = repmat(gobjects(1), maxBoxes, 1);
            this.ParcelTextHandles = repmat(gobjects(1), maxBoxes, 1);
            
            colors = lines(max(maxBoxes,1));
            for kk = 1:maxBoxes
                % Creiamo i rettangoli "invisibili" pronti all'uso
                this.ParcelRectHandles(kk) = rectangle(ha, 'Position', [0 0 0 0], ...
                    'FaceColor', colors(kk,:), 'EdgeColor', [0 0 0], 'LineWidth', 1.0, 'Visible', 'off');
                
                % Creiamo i testi dei numeri identificativi "invisibili"
                this.ParcelTextHandles(kk) = text(ha, 0, 0, num2str(kk), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'FontSize', 8, 'FontWeight', 'bold', 'Color', [1 1 1], 'Visible', 'off');
            end
        end
        
        function updatePlot(this)
            env = this.Environment;
            f = this.Figure;
            axList = findobj(f,'Type','axes');
            if isempty(axList), return; end
            ha = axList(1);
            
            yExit = env.d_AMS*env.n_i_AMS;
            yEval = yExit + 1.0;
            
            % Reset valutazione ad inizio episodio
            if isempty(this.EvalTopRecorded) || ...
                    numel(this.EvalTopRecorded) ~= env.max_gen_boxes || ...
                    env.cont <= 1 || env.cont < this.LastCont
                this.resetEvaluation(env.max_gen_boxes);
            end
            this.LastCont = env.cont;
            [evalAccuracy,nPairs,lastGap] = this.updateEvaluation(env,yEval);
            
            % -----------------------------------------------------------------
            % AGGIORNAMENTO VELOCE DELLE PROPRIETÀ (NO DELETE, NO ALLOCAZIONE)
            % -----------------------------------------------------------------
            
            % Aggiorna testo velocità nastro
            set(this.TreadmillTextHandle, 'String', sprintf('outfeed treadmill: v = %.2f m/s', env.v_treadmill));
            
            % Aggiorna i pacchi (Parcels)
            nBoxesAvailable = numel(env.x_boxes);
            for kk = 1:env.max_gen_boxes
                if kk <= nBoxesAvailable && env.x_boxes(kk) >= 0 && env.y_boxes(kk) >= 0 && env.d_boxes(kk) > 0
                    d = env.d_boxes(kk);
                    xLeft = env.x_boxes(kk) - d/2;
                    yBottom = env.y_boxes(kk) - d/2;
                    
                    % Modifica al volo posizione e rendi visibile
                    set(this.ParcelRectHandles(kk), 'Position', [xLeft yBottom d d], 'Visible', 'on');
                    set(this.ParcelTextHandles(kk), 'Position', [env.x_boxes(kk) env.y_boxes(kk) 0], 'Visible', 'on');
                else
                    % Nascondi se il pacco non è ancora attivo o è uscito dalla matrice
                    set(this.ParcelRectHandles(kk), 'Visible', 'off');
                    set(this.ParcelTextHandles(kk), 'Visible', 'off');
                end
            end
            
            % Aggiorna i pannelli di testo delle informazioni
            if isnan(evalAccuracy), accText = 'n/a'; else, accText = sprintf('%.1f%%',evalAccuracy); end
            if isnan(lastGap), gapText = 'n/a'; else, gapText = sprintf('%.3f m',lastGap); end
            
            info1 = sprintf('t = %.2f s | generated = %d/%d | evaluated pairs = %d', ...
                env.time,env.n_boxes_tot,env.max_gen_boxes,nPairs);
            info2 = sprintf('accuracy @ yExit + 1 m: %s | last gap = %s', ...
                accText,gapText);
            
            set(this.InfoTextHandle1, 'String', info1);
            set(this.InfoTextHandle2, 'String', info2);
            
            % Limitazione del refresh rate grafico per non impattare sul calcolo di RL
            drawnow limitrate;
        end
    end
    methods (Access = private)
        function resetEvaluation(this,maxBoxes)
            this.EvalTopRecorded = false(maxBoxes,1);
            this.EvalOrder = zeros(0,1);
            this.EvalGaps = zeros(0,1);
            this.EvalSuccessCount = 0;
            this.LastCont = -1;
        end
        function [accuracy,nPairs,lastGap] = updateEvaluation(this,env,yEval)
            nBoxes = min(env.n_boxes_tot,env.max_gen_boxes);
            newIdx = zeros(0,1);
            newTop = zeros(0,1);
            for kk = 1:nBoxes
                if kk > numel(this.EvalTopRecorded) || this.EvalTopRecorded(kk)
                    continue;
                end
                if env.x_boxes(kk) < 0 || env.y_boxes(kk) < 0 || env.d_boxes(kk) <= 0
                    continue;
                end
                yTop = env.y_boxes(kk) + env.d_boxes(kk)/2;
                if yTop >= yEval
                    newIdx(end+1,1) = kk; %#ok<AGROW>
                    newTop(end+1,1) = yTop; %#ok<AGROW>
                end
            end
            if ~isempty(newIdx)
                [~,ord] = sort(newTop,'descend');
                newIdx = newIdx(ord);
            end
            for ii = 1:numel(newIdx)
                curr = newIdx(ii);
                if this.EvalTopRecorded(curr)
                    continue;
                end
                this.EvalTopRecorded(curr) = true;
                if ~isempty(this.EvalOrder)
                    prev = this.EvalOrder(end);
                    bottomPrevious = env.y_boxes(prev) - env.d_boxes(prev)/2;
                    topCurrent = env.y_boxes(curr) + env.d_boxes(curr)/2;
                    gap = bottomPrevious - topCurrent;
                    this.EvalGaps(end+1,1) = gap; %#ok<AGROW>
                    if gap > 0
                        this.EvalSuccessCount = this.EvalSuccessCount + 1;
                    end
                end
                this.EvalOrder(end+1,1) = curr; %#ok<AGROW>
            end
            nPairs = numel(this.EvalGaps);
            if nPairs == 0
                accuracy = NaN;
                lastGap = NaN;
            else
                accuracy = 100*this.EvalSuccessCount/nPairs;
                lastGap = this.EvalGaps(end);
            end
        end
    end
end