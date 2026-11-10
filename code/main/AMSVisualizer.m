classdef AMSVisualizer < rl.env.viz.AbstractFigureVisualizer
    %AMSVISUALIZER Visualizer for the AMS sorting environment.
    %
    % Generated from AMSVisualizer_simple, with actuator arrows added.
    %
    % The plot limits are fixed to the physical 2 m region:
    %   y = 0...1 m : AMS sorting matrix / singulator
    %   y = 1...2 m : outfeed treadmill
    %
    % It shows:
    %   - AMS actuator matrix;
    %   - outfeed treadmill;
    %   - AMS exit line;
    %   - evaluation line 1 m after the AMS exit;
    %   - actuator arrows indicating action orientation and velocity module;
    %   - parcels with correct physical size;
    %   - cumulative sorting accuracy at the evaluation line.
    %
    % Accuracy is computed at:
    %
    %   yEval = yExit + 1.0
    %
    % For two consecutive parcels in the output stream:
    %
    %   gap = bottomPrevious - topCurrent
    %
    % where:
    %
    %   bottomPrevious = y_previous - d_previous/2
    %   topCurrent     = y_current  + d_current/2
    %
    % A pair is counted as successfully singulated when gap > 0.

    properties (Access = private)
        EvalTopRecorded = []
        EvalOrder = []
        EvalGaps = []
        EvalSuccessCount = 0
        LastCont = -1
    end

    methods
        function this = AMSVisualizer(env)
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
                'Name','AMS sorting visualizer', ...
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

            ha.XLim = [0 xMax];
            ha.YLim = [0 yEval];

            xlabel(ha,'x [m]');
            ylabel(ha,'y [m]');
            title(ha,'AMS sorting environment');
        end

        function updatePlot(this)
            env = this.Environment;
            f = this.Figure;

            axList = findobj(f,'Type','axes');
            if isempty(axList)
                return;
            end
            ha = axList(1);

            dAMS = env.d_AMS;
            xMax = env.d_AMS*env.n_j_AMS;
            yExit = env.d_AMS*env.n_i_AMS;
            yEval = yExit + 1.0;

            % Reset visualizer-side evaluation at the start of a new episode.
            if isempty(this.EvalTopRecorded) || ...
                    numel(this.EvalTopRecorded) ~= env.max_gen_boxes || ...
                    env.cont <= 1 || env.cont < this.LastCont
                this.resetEvaluation(env.max_gen_boxes);
            end
            this.LastCont = env.cont;

            [evalAccuracy,nPairs,lastGap] = this.updateEvaluation(env,yEval);

            delete(findobj(ha,'Tag','ams_static'));
            delete(findobj(ha,'Tag','ams_dynamic'));
            delete(findobj(ha,'Tag','ams_text'));

            % Fixed physical view: 1 m AMS matrix + 1 m outfeed treadmill.
            % Do not expand the plot when parcels go beyond the evaluation line.
            ha.XLim = [0 xMax];
            ha.YLim = [0 yEval];

            % -----------------------------------------------------------------
            % Layout: AMS matrix and outfeed treadmill.
            % -----------------------------------------------------------------
            rectangle(ha,'Position',[0 yExit xMax 1.0], ...
                'FaceColor',[0.92 0.92 0.92], ...
                'EdgeColor','none', ...
                'Tag','ams_static');

            rectangle(ha,'Position',[0 0 xMax yExit], ...
                'FaceColor',[1 1 1], ...
                'EdgeColor',[0 0 0], ...
                'LineWidth',1.5, ...
                'Tag','ams_static');

            % AMS grid.
            for ii = 0:env.n_i_AMS
                line(ha,[0 xMax],[ii*dAMS ii*dAMS], ...
                    'LineWidth',1.0, ...
                    'Color',[0 0 0], ...
                    'Tag','ams_static');
            end
            for jj = 0:env.n_j_AMS
                line(ha,[jj*dAMS jj*dAMS],[0 yExit], ...
                    'LineWidth',1.0, ...
                    'Color',[0 0 0], ...
                    'Tag','ams_static');
            end

            % AMS exit line.
            line(ha,[0 xMax],[yExit yExit], ...
                'LineWidth',2.5, ...
                'LineStyle','--', ...
                'Color',[0.20 0.20 0.20], ...
                'Tag','ams_static');

            % Evaluation line: 1 m after AMS exit.
            line(ha,[0 xMax],[yEval yEval], ...
                'LineWidth',3.0, ...
                'LineStyle','-', ...
                'Color',[0.85 0.10 0.10], ...
                'Tag','ams_static');

            text(ha,0.02,yExit+0.03,'AMS exit', ...
                'FontSize',9, ...
                'FontWeight','bold', ...
                'Color',[0.20 0.20 0.20], ...
                'Tag','ams_text');

            text(ha,0.02,yEval-0.05,'accuracy evaluation line: 1 m after AMS exit', ...
                'FontSize',9, ...
                'FontWeight','bold', ...
                'Color',[0.85 0.10 0.10], ...
                'Tag','ams_text');

            treadmillLabel = sprintf('outfeed treadmill: v = %.2f m/s',env.v_treadmill);
            text(ha,0.02,yExit+0.50,treadmillLabel, ...
                'FontSize',9, ...
                'Color',[0.25 0.25 0.25], ...
                'Tag','ams_text');

            % Treadmill direction indication.
            for xx = linspace(0.20,xMax-0.20,3)
                quiver(ha,xx,yExit+0.15,0,0.16,0, ...
                    'LineWidth',1.0, ...
                    'MaxHeadSize',1.5, ...
                    'Color',[0.35 0.35 0.35], ...
                    'Tag','ams_static');
            end

            % -----------------------------------------------------------------
            % Actuator arrows: orientation = rotation action, module = velocity.
            %
            % Convention used by RL_environment:
            %   rotation = 0  -> forward along +y
            %   dx = arrowLength*sin(rotation)
            %   dy = arrowLength*cos(rotation)
            %
            % LastAction is expected to contain physical actions:
            %   [rot_1, vel_1, rot_2, vel_2, ...]
            % in row-major order.
            % -----------------------------------------------------------------
            action = env.LastAction;
            nExpectedActions = env.n_i_AMS*env.n_j_AMS*env.n_actions_art;

            if numel(action) == nExpectedActions
                action = action(:);

                for rr = 1:env.n_i_AMS
                    for cc = 1:env.n_j_AMS
                        idx = (rr-1)*env.n_j_AMS*env.n_actions_art + ...
                              (cc-1)*env.n_actions_art + 1;

                        rot = action(idx);
                        vel = action(idx+1);

                        velNorm = (vel-env.lowerlim_v)/(env.upperlim_v-env.lowerlim_v);
                        velNorm = max(0,min(1,velNorm));

                        % Length and line width both encode the velocity module.
                        arrowLength = 0.035 + 0.095*velNorm;
                        arrowWidth = 0.8 + 1.6*velNorm;

                        x0 = (cc-0.5)*dAMS;
                        y0 = (rr-0.5)*dAMS;

                        dx = arrowLength*sin(rot);
                        dy = arrowLength*cos(rot);

                        quiver(ha,x0,y0,dx,dy,0, ...
                            'LineWidth',arrowWidth, ...
                            'MaxHeadSize',2.2, ...
                            'Color',[0.05 0.05 0.05], ...
                            'Tag','ams_dynamic');

                        % Small center marker to make the arrow origin clear.
                        plot(ha,x0,y0,'.', ...
                            'MarkerSize',7, ...
                            'Color',[0.05 0.05 0.05], ...
                            'Tag','ams_dynamic');
                    end
                end
            end

            % -----------------------------------------------------------------
            % Parcels.
            % x_boxes/y_boxes are centroids. MATLAB rectangle uses lower-left.
            % -----------------------------------------------------------------
            colors = lines(max(env.max_gen_boxes,1));

            for kk = 1:min(env.max_gen_boxes,numel(env.x_boxes))
                if env.x_boxes(kk) >= 0 && env.y_boxes(kk) >= 0 && env.d_boxes(kk) > 0
                    d = env.d_boxes(kk);
                    xLeft = env.x_boxes(kk) - d/2;
                    yBottom = env.y_boxes(kk) - d/2;

                    rectangle(ha,'Position',[xLeft yBottom d d], ...
                        'FaceColor',colors(kk,:), ...
                        'EdgeColor',[0 0 0], ...
                        'LineWidth',1.0, ...
                        'Tag','ams_dynamic');

                    text(ha,env.x_boxes(kk),env.y_boxes(kk),num2str(kk), ...
                        'HorizontalAlignment','center', ...
                        'VerticalAlignment','middle', ...
                        'FontSize',8, ...
                        'FontWeight','bold', ...
                        'Color',[1 1 1], ...
                        'Tag','ams_dynamic');
                end
            end

            % -----------------------------------------------------------------
            % Text panel.
            % -----------------------------------------------------------------
            if isnan(evalAccuracy)
                accText = 'n/a';
            else
                accText = sprintf('%.1f%%',evalAccuracy);
            end

            if isnan(lastGap)
                gapText = 'n/a';
            else
                gapText = sprintf('%.3f m',lastGap);
            end

            info1 = sprintf('t = %.2f s | generated = %d/%d | evaluated pairs = %d', ...
                env.time,env.n_boxes_tot,env.max_gen_boxes,nPairs);

            info2 = sprintf('accuracy @ yExit + 1 m: %s | last gap = %s', ...
                accText,gapText);

            text(ha,0.02,yEval-0.18,info1, ...
                'FontSize',10, ...
                'FontWeight','bold', ...
                'BackgroundColor',[1 1 1], ...
                'Margin',3, ...
                'Tag','ams_text');

            text(ha,0.02,yEval-0.28,info2, ...
                'FontSize',10, ...
                'FontWeight','bold', ...
                'BackgroundColor',[1 1 1], ...
                'Margin',3, ...
                'Tag','ams_text');

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
            % A parcel enters the output stream when its top/front edge reaches
            % the evaluation line.
            %
            % The gap is:
            %
            %   bottom of previous parcel - top of current parcel
            %
            % gap > 0 means a successful longitudinal separation.

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

            % If several parcels cross in the same frame, the one further
            % downstream is considered first.
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
