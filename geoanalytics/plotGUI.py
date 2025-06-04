import math
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class plotGUI:
    colorOptions = ['#a3a7e4', '#000000']
    sizeOptions = [10, 15]
    exportReady = pd.DataFrame()
    
    def drawEnvelope(dataframe):
        figuresEnvelope = go.FigureWidget(
            make_subplots(rows=2, cols=1, subplot_titles=("Click on point for band value information", "Values")))
        
        figuresEnvelope.update_layout(autosize=False,width=800,height=800)
        
        indexHeader = {"Values" : dataframe.columns[2:]}
        
        clicked = list()
        clickedNames = list()
        clickedDataframe = pd.DataFrame(indexHeader)
        clickedDataframe = clickedDataframe.set_index("Values")
        
        def updatePoint(trace,points,selector):
            pointIndex = points.point_inds[0]
            size = list(plotPoints.marker.size)
            pointData = dataframe.loc[pointIndex].tolist()
            point = str(pointData[0]) + " " + str(pointData[1])
            #if already clicked
            if point in clickedNames:
                clickedNames.remove(point)
                clickedDataframe.drop(columns=point, inplace=True)
                for i in points.point_inds:
                    size[i] = plotGUI.sizeOptions[0]
                    with figuresEnvelope.batch_update():
                        plotPoints.marker.size = size
            #if not clicked
            else:
                clickedDataframe[point] = pointData[2:]
                clickedNames.append(point)
                for i in points.point_inds:
                    size[i] = plotGUI.sizeOptions[1]
                    with figuresEnvelope.batch_update():
                        plotPoints.marker.size = size
                        
            #reset plot data
            figuresEnvelope.data = figuresEnvelope.data[:len(figuresEnvelope.data) - (len(figuresEnvelope.data) - 1)]
            
            #redraw plot
            for i in range(len(clickedDataframe.columns)):
                figuresEnvelope.add_trace(go.Scatter(x=clickedDataframe.loc[:, clickedDataframe.columns[i]].index,
                                                     y=clickedDataframe.loc[:, clickedDataframe.columns[i]],
                                                     mode='markers+lines', name=clickedNames[i], ), row=2, col=1)
            plotGUI.exportReady = clickedDataframe
        
        #add first plot to click
        figuresEnvelope.add_trace(
            go.Scatter(x=dataframe.x, y=dataframe.y, mode='markers', showlegend=False))
        
        
        plotPoints = figuresEnvelope.data[0]
        plotPoints.marker.color = [plotGUI.colorOptions[0]] * len(dataframe)
        plotPoints.marker.size = [plotGUI.sizeOptions[0]] * len(dataframe)
        plotPoints.on_click(updatePoint)
        display(figuresEnvelope)
    
    
    def drawNearestPoint(dataframe):
        figuresNearestPoint = go.FigureWidget(
            make_subplots(rows=2, cols=1, subplot_titles=("Click on point for band value information", "Values")))
        
        figuresNearestPoint.update_layout(autosize=False,width=800,height=800)
        
        indexHeader = {"Values" : dataframe.columns[2:]}
        
        clicked = list()
        clickedNames = list()
        clickedDataframe = pd.Dataframe(indexHeader)
        clickedDataframe = clickedDataframe.set_index("Values")
        
        def updatePoint(trace,points,selector):
            pointIndex = points.point_inds[0]
            size = list(plotPoints.marker.size)
            pointData = dataframe.loc[pointIndex].tolist()
            point = str(pointData[0]) + " " + str(pointData[1])
            #if already clicked
            if point in clickedNames:
                clickedNames.remove(point)
                clickedDataframe.drop(columns=point, inplace=True)
                for i in points.point_inds:
                    size[i] = plotGUI.sizeOptions[0]
                    with figuresNearestPoint.batch_update():
                        plotPoints.marker.size = size
            #if not clicked
            else:
                clickedDataframe[point] = pointData[2:]
                clickedNames.append(point)
                for i in points.point_inds:
                    size[i] = plotGUI.sizeOptions[1]
                    with figuresNearestPoint.batch_update():
                        plotPoints.marker.size = size
                        
            #reset plot data
            figuresNearestPoint.data = figuresNearestPoint.data[:len(figuresNearestPoint.data) -
                                                                (len(figuresNearestPoint.data) - 1)]
            
            #redraw plot
            for i in range(len(clickedDataframe.columns)):
                figuresNearestPoint.add_trace(go.Scatter(x=clickedDataframe.loc[:, clickedDataframe.columns[i]].index,
                                                         y=clickedDataframe.loc[:, clickedDataframe.columns[i]],
                                                         mode='markers+lines', name=clickedNames[i], ), row=2, col=1)
            plotGUI.exportReady = clickedDataframe
        
        #add first plot to click
        figuresNearestPoint.add_trace(
            go.Scatter(x=dataframe.x, y=dataframe.y, mode='markers', showlegend=False), row=1, col=1)
        
        #color the points
        distances = list()
        
        xPos = dataframe['x'][0]
        yPos = dataframe['y'][0]
        for i in range(len(dataframe)):
            distance = math.sqrt((abs(xPos - dataframe['x'][i]) *
                                  abs(xPos - dataframe['x'][i])) +
                                 (abs(yPos - dataframe['y'][i]) *
                                  abs(yPos - dataframe['y'][i])))
            distances.append(distance)
        
        def rgbhex(r, g, b):
            def clamp(x):
                return max(0, min(x, 255))
            return '#{0:02x}{1:02x}{2:02x}'.format(clamp(r), clamp(g), clamp(b))
        
        distanceColor = list()
        for i in range(len(dataframe)):
            value = int(distances[i] * max(300,300/len(dataframe)*300))
            distanceColor.append(rgbhex(255 - value, value, value))
        
        distanceColor[0] = plotGUI.colorOptions[1]
        sizeList = [plotGUI.sizeOptions[0]] * (len(dataframe))
        sizeList[0] = plotGUI.sizeOptions[1]

        plotPoints = figuresKNN.data[0]
        plotPoints.marker.color = tuple(distanceColor)
        plotPoints.marker.size = tuple(sizeList)
        
        figuresKNN.add_trace(go.Scatter(x=dataframe.iloc[0][2:].index, 
                                        y=dataframe.iloc[0][2:], mode='markers+lines',
                                        name=str(dataframe.iloc[0][0]) + " " + 
                                        str(dataframe.iloc[0][1]), ), row=2, col=1)
        
        plotPoints = figuresNearestPoint.data[0]
        
        clickedDataframe[str(dataframe.iloc[0][0]) + " " +
                  str(dataframe.iloc[0][1])] = dataframe.iloc[0]
        clickedNames.append(str(dataframe.iloc[0][0]) + " " + str(dataframe.iloc[0][1]))
        
        dataframe.drop([0],inplace=True)
        
        plotPoints.on_click(updatePoint)
        display(figuresNearestPoint)